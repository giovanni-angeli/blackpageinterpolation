# coding: utf-8

# pylint: disable=missing-docstring
# pylint: disable=logging-format-interpolation
# pylint: disable=line-too-long
# pylint: disable=invalid-name
# pylint: disable=too-many-lines

import sys
import os
import time
import asyncio
import logging
import traceback
import json
import random

import tornado.web            # pylint: disable=import-error
import tornado.httpserver     # pylint: disable=import-error
import tornado.ioloop         # pylint: disable=import-error
import tornado.websocket      # pylint: disable=import-error
import tornado.options        # pylint: disable=import-error

HERE = os.path.dirname(os.path.abspath(__file__))

LISTEN_PORT = 8000
LISTEN_ADDRESS = '127.0.0.1'

N_OF_WORKERS = 10

APPLICATION_OPTIONS = dict(
    debug=True,
    autoreload=True,
    template_path=os.path.join(HERE, "..", "templates"),
    compiled_template_cache=False)

GLOBAL_APPLICATION_INSTANCE = None


def get_application_instance():

    global GLOBAL_APPLICATION_INSTANCE  # pylint: disable=global-statement

    if GLOBAL_APPLICATION_INSTANCE is None:
        GLOBAL_APPLICATION_INSTANCE = Application()

    return GLOBAL_APPLICATION_INSTANCE


class HttpHandler(tornado.web.RequestHandler):       # pylint: disable=too-few-public-methods

    def get(self):

        ctx = {
            "title": "Po‐Jama People",
            "footer": "© Munchkin Music Co - https://www.zappa.com/",
        }
        ret = self.render("index.html", **ctx)
        return ret


class WebsockHandler(tornado.websocket.WebSocketHandler):

    def initialize(self):

        a = get_application_instance()
        a.web_socket_channels.append(self)
        logging.info(f"n. of active web_socket_channels:{len(a.web_socket_channels)}")

    def open(self, *args, **kwargs):

        super().open(args, kwargs)
        logging.info(f"")

    def on_message(self, message):

        a = get_application_instance()
        t_ = a.handle_message_from_UI(self, message)
        asyncio.ensure_future(t_)

    def on_close(self):

        a = get_application_instance()
        a.web_socket_channels.remove(self)
        logging.info(f"n. of active web_socket_channels:{len(a.web_socket_channels)}")


class Worker:       # pylint: disable=too-few-public-methods

    def __init__(self, id_):

        self.id = id_

    async def run(self):

        logging.info(f"START self.id :{self.id }")

        app_ = get_application_instance()

        while True:

            try:
                app_.send_message_to_UI("time_display", time.asctime())
                app_.send_message_to_UI("channel_counter", len(get_application_instance().web_socket_channels))

                for i in range(1, 4):
                    for j in range(1, 6):
                        _spam = await app_.get_from_single_lockable_resource(self.id)
                        app_.send_message_to_UI(f"datatable_cell_{i}_{j}", "id :{} {}".format(self.id, _spam))

            except tornado.websocket.WebSocketClosedError:
                pass

            except Exception:    # pylint: disable=broad-except
                logging.error(traceback.format_exc())

            await asyncio.sleep(1)

        logging.info(f"EXIT self.id :{self.id }")


class Application:

    url_map = [
        (r"/", HttpHandler, {}),
        (r'/websocket', WebsockHandler, {}),
    ]

    web_socket_channels = []
    who_is_locking = []
    waiting_worker_ids = []

    running_workers = {}

    async def wait_for_condition(self,  # pylint: disable=too-many-arguments
                                 condition, extra_info=None, timeout=10, stability_count=2, step=0.1):

        ret = None
        t0 = time.time()
        counter = 0
        try:
            while time.time() - t0 < timeout:

                if condition and condition():
                    counter += 1
                    if counter >= stability_count:
                        ret = True
                        break
                else:
                    counter = 0

                await asyncio.sleep(step)

            if not ret:
                _ = f"timeout expired! timeout:{timeout}.\n"
                if extra_info:
                    _ += str(extra_info)
                logging.info(_)

        except Exception:  # pylint: disable=broad-except
            logging.error(traceback.format_exc())

        return ret

    async def get_from_single_lockable_resource(self, task_id):

        ret = None

        try:

            def condition():
                return not self.who_is_locking

            self.waiting_worker_ids.append(task_id)
            r = await self.wait_for_condition(condition, timeout=10, step=.001, extra_info=f"task_id:{task_id} ")
            self.waiting_worker_ids.remove(task_id)

            if r:
                self.who_is_locking.append(task_id)
                logging.debug(
                    f"who_is_locking:{str(self.who_is_locking).ljust(20)}, waiting_worker_ids:{self.waiting_worker_ids}")
                assert len(self.who_is_locking) == 1
                assert self.who_is_locking not in self.waiting_worker_ids

                await asyncio.sleep(.001)
                ret = content_producer()
                await asyncio.sleep(.001)

                self.who_is_locking.remove(task_id)
                logging.debug(
                    f"who_is_locking:{str(self.who_is_locking).ljust(20)}, waiting_worker_ids:{self.waiting_worker_ids}")
                assert len(self.who_is_locking) == 0

                await asyncio.sleep(.00001)

        except Exception:    # pylint: disable=broad-except
            logging.error(traceback.format_exc())

        return ret

    async def handle_message_from_UI(self, ws_socket, message):

        index_ = self.web_socket_channels.index(ws_socket)

        logging.info(f"index_:{index_}, message:{message}")

        answer = await self.get_from_single_lockable_resource(-1)

        innerHTML = "ws_index:{} [{}] {} -> {}".format(
            index_, time.asctime(), message, answer)

        self.send_message_to_UI("answer_display", innerHTML)

    def send_message_to_UI(self, element_id, innerHTML, ws_index=None):

        msg = {"element_id": element_id, "innerHTML": innerHTML}
        msg = json.dumps(msg)

        if ws_index:
            t_ = self.web_socket_channels[ws_index].write_message(msg)
            asyncio.ensure_future(t_)

        else:  # broadcast
            for ws_ch in self.web_socket_channels:
                t_ = ws_ch.write_message(msg)
                asyncio.ensure_future(t_)

    def start_tornado(self):

        logging.info("starting tornado webserver on http://{}:{}...".format(LISTEN_ADDRESS, LISTEN_PORT))

        app = tornado.web.Application(self.url_map, **APPLICATION_OPTIONS)
        app.listen(LISTEN_PORT, LISTEN_ADDRESS)
        tornado.platform.asyncio.AsyncIOMainLoop().install()

    def start_backend_workers(self, n_of_workers):

        logging.info(f"starting {n_of_workers} backend workers...")

        for i in range(n_of_workers):
            _w = Worker(id_=i)
            _t = _w.run()
            self.running_workers[i] = (_w, _t)
            asyncio.ensure_future(_t)

    def run(self, n_of_workers):

        self.start_tornado()
        self.start_backend_workers(n_of_workers)

        asyncio.get_event_loop().run_forever()


def main():

    logging.basicConfig(
        stream=sys.stdout,
        level="INFO",
        # ~ level="DEBUG",
        format="[%(asctime)s]%(levelname)s %(funcName)s() %(filename)s:%(lineno)d %(message)s")

    a = get_application_instance()
    a.run(n_of_workers=N_OF_WORKERS)


def content_producer():
    """
Fonte: Musixmatch
Compositori: Zappa F
Testo di Po‐Jama People © Munchkin Music Co
    """

    _ = """
Some people's hot
Some people's cold
Some people's not very
Swift to behold
Some people do it
Some see right through it
Some wear pyjamas
If only they knew it
The pyjama people are boring me to pieces
Feel like I am wasting my time
They all got flannel up 'n down 'em
A little trap-door back aroun' 'em
An' some cozy little footies on their mind
Po-jama people!
Po-jama people, people!
They sure do make you sleepy
With the things they might say
Po-jama people!
Po-jama people, people!
Mother Mary 'n Jozuf, I wish they'd all go away!
Po-jama people!
It's a po-jama people special . . .
Take one home with you, save a dollar today
Po-jama people!
Po-jama people, people!
Wrap 'em up
Roll 'em out
Get 'em out of my way
Hein nya-nya-hein nya-nya-hein nya-nya-hein
HOEY! HOEY! HOEY!
Wrap 'em up
Roll 'em out
Get 'em out of my way
Hein nya-nya-hein nya-nya-hein nya-nya-hein
HOEY! HOEY! HOEY!
Wrap 'em up
Roll 'em out
Get 'em out of my way
Hein nya-nya-hein nya-nya-hein nya-nya-hein
HOEY! HOEY! HOEY!
now some people's hot
An' some people's cold
(Well, Lawd . . .) an' some people's not very
(Very) swift to behold (swifty!)
(I told you) some people do it (do it!)
(Yes, they do!) (No . . .)
Some see right through it
(See right through it!)
An' some wear PO-JAMAS
If only they knew it
The pyjama people are boring me to pieces
They make me feel like I am wasting my time
They all got flannel up 'n down 'em
A little trap-door back aroun' 'em
An' some cozy little footies on their mind
Po-jama people!
Po-jama people, people!
Lawd, they make you sleepy
With the things they might say (hey, yeah-hey . . .)
Po-jama people!
(Well . . . now) Po-jama people, people!
(I said) ARF! ARF! ARF!
I wish they'd all go away!
Po-jama people! (People!) (Oh, yeah)
Po-jama people special . . .
(I said, I said, I said)
Take one home with you, & save a dollar today
Po-jama people!
(It's a) Po-jama people, people! (Special)
Wrap 'em up
An' roll 'em out
Get 'em out of my way
HOEY! HOEY! HOEY!
Wrap 'em up
An' roll 'em out
Get 'em out of my way
HOEY! HOEY! HOEY!
Wrap 'em up
Roll 'em out
Get 'em out of my way
HOEY! HOEY! HOEY!
"""

    return random.choice([t for t in _.split('\n') if t])


if __name__ == '__main__':
    main()
