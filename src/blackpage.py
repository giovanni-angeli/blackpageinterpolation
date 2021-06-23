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
# ~ import random

import tornado.web            # pylint: disable=import-error
import tornado.httpserver     # pylint: disable=import-error
import tornado.ioloop         # pylint: disable=import-error
import tornado.websocket      # pylint: disable=import-error
import tornado.options        # pylint: disable=import-error
import tornado.escape

from rbf_interpolator import Model

HERE = os.path.dirname(os.path.abspath(__file__))

LISTEN_PORT = 8000
LISTEN_ADDRESS = '127.0.0.1'

APPLICATION_OPTIONS = dict(
    debug=True,
    autoreload=True,
    template_path=os.path.join(HERE, 'templates'),
    static_path=os.path.join(HERE, 'statics'),
    compiled_template_cache=False)


class HttpHandler(tornado.web.RequestHandler): # pylint: disable=too-few-public-methods, abstract-method

    parent = None

    def initialize(self, *args, **kwargs):

        self.parent = kwargs.pop('parent')
        super().initialize(*args, **kwargs)

    def get(self):

        ctx = {
            "title": "ColorInterpolator",
            "footer": "© Munchkin Music Co - https://www.zappa.com/",
            # ~ "params_panel": tornado.escape.xhtml_escape(self.parent.render_params_html())
            "params_panel": self.parent.render_params_html()
        }
        ret = self.render("index.html", **ctx)
        return ret


class WebsockHandler(tornado.websocket.WebSocketHandler): # pylint: disable=abstract-method

    parent = None

    def initialize(self, **kwargs):

        self.parent = kwargs.pop('parent')
        self.parent.web_socket_channels.append(self)

    def open(self, *args, **kwargs):

        super().open(*args, **kwargs)
        logging.info(f"")

    def on_message(self, message):

        if self.parent:
            t_ = self.parent.handle_message_from_UI(self, message)
            asyncio.ensure_future(t_)

    def on_close(self):

        if self.parent:
            self.parent.web_socket_channels.remove(self)
            logging.info(f"n. of active web_socket_channels:{len(self.parent.web_socket_channels)}")


class Backend:       # pylint: disable=too-few-public-methods

    def __init__(self, parent=None):

        pigment_combos = [
            "jauox,noir,rouox",
            "jaune,jauox,vert",
            "jauox,noir,vert",
            "jaune,orange,vert",
            "bleu,noir,vert",
            "bleu,orange,vert",
            "bleu,jauox,vert",
            "jauox,reddish yellow,vert",
            "reddish yellow,rouox,vert",
            "jaune,jauox,orange",
            "bleu,noir,violet",
            "bleu,jauox,noir",
            "bleu,orange,violet",
        ]

        self.parent = parent
        self.params = dict(
            rfb_name='test',
            epsilon=1.0,
            n_of_rotate_sample=10,
            n_of_sites=100,
            mockup=False,
            pigment_combo=pigment_combos[0],
            offset=0)

        self.param_descriptions = dict(
            pigment_combo=f"{pigment_combos}")

        self.model = None
        self.results = None

    def update_progress(self, progress):

        self.parent.send_message_to_UI(element_id='answer_display', innerHTML=progress)

    def run_model(self):

        data_dir = "/opt/PROJECTS/blackpageinterpolation/_ignore_/data/recipes"

        try:
            _params = self.params.copy()
            _params['epsilon'] = float(self.params['epsilon'])
            _params['n_of_rotate_sample'] = int(self.params['n_of_rotate_sample'])
            _params['n_of_sites'] = int(self.params['n_of_sites'])
            _params['mockup'] = False
            _params['f_name'] = os.path.join(data_dir, "recipes.Master base,Neutro," + self.params['pigment_combo'] + ",white.json")
            _params['offset'] = int(self.params['offset'])

            logging.info(f"_params:{_params}")

            self.model = Model(**_params)

            self.model.progress = self.update_progress

            self.results = self.model.run()

        except Exception:  # pylint: disable=broad-except
            self.results = traceback.format_exc()
            logging.error(traceback.format_exc())

        _, html_body, _ = self.model.render_results_html(self.results, title="")
        return html_body

    def store_results(self, ):

        data_dir = "/opt/PROJECTS/blackpageinterpolation/_ignore_/data/"

        self.results.sort(key=lambda x: x.get('error', 0) and x['error'])
        f_name = f"results_{self.params['n_of_sites']}_{self.params['n_of_rotate_sample']}_{self.params['pigment_combo']}.json"
        with open(os.path.join(data_dir, f_name), 'w') as f:
            json.dump(self.results, f, indent=2)

        return f_name

    def refresh_results(self, order_by='rgb'):

        if order_by.lower() == 'r':
            self.results.sort(key=lambda x: x.get('rgb', 0) and x['rgb'][0])
        elif order_by.lower() == 'g':
            self.results.sort(key=lambda x: x.get('rgb', 0) and x['rgb'][1])
        elif order_by.lower() == 'b':
            self.results.sort(key=lambda x: x.get('rgb', 0) and x['rgb'][2])
        else:
            self.results.sort(key=lambda x: x.get(order_by) and x[order_by])

        _, html_body, _ = self.model.render_results_html(self.results, title="")
        return html_body

    async def run(self):

        logging.info(f"START")

        while True:
            try:
                self.parent.send_message_to_UI("time_display", time.asctime())
                self.parent.send_message_to_UI("channel_counter", len(self.parent.web_socket_channels))
            except tornado.websocket.WebSocketClosedError as e:
                logging.info(f"e:{e}")
            except Exception:    # pylint: disable=broad-except
                logging.error(traceback.format_exc())

            await asyncio.sleep(1)

        logging.info(f"EXIT")


class Application:

    web_socket_channels = []
    tornado_instance = None
    backend_instance = None

    def render_params_html(self):

        html_ = '<table>'
        for k, v in self.backend_instance.params.items():
            description = self.backend_instance.param_descriptions.get(k)
            html_ += f'<tr>'
            html_ += f'<td style="text-align:right;width:10%;">{k}:</td>'
            html_ += f'<td style="text-align:left;"><input value="{v}" name="{k}" size="40" class="params_panel_item"></input></td>'
            html_ += f'<td style="width:50%;">{description}</td>'
            html_ += f'</tr>'
        html_ += '</table>'

        return html_

    async def handle_message_from_UI(self, ws_socket, pack):

        index_ = ws_socket in self.web_socket_channels and self.web_socket_channels.index(ws_socket)
        pack = json.loads(pack)
        logging.info(f"index_:{index_}, pack:{pack}")
        if pack.get('message') == 'run_model':

            self.send_message_to_UI(element_id='answer_display', innerHTML='Running Model, please wait...', ws_socket=ws_socket)
            results_html = self.backend_instance.run_model()
            self.send_message_to_UI(element_id='data_display', innerHTML=results_html, ws_socket=ws_socket)

        elif pack.get('message') == 'store_results':

            f_name = self.backend_instance.store_results()
            self.send_message_to_UI(element_id='answer_display', innerHTML=f'stored data to "{f_name}"', ws_socket=ws_socket)

        elif pack.get('message') == 'refresh_results':

            order_by = pack.get('option', 'rgb')
            results_html = self.backend_instance.refresh_results(order_by=order_by)
            self.send_message_to_UI(element_id='data_display', innerHTML=results_html, ws_socket=ws_socket)

        elif pack.get('message') == 'stop_model':

            if self.backend_instance.model:
                self.backend_instance.model.stop()

        elif pack.get('message') == 'params_panel':

            params = pack.get('option')
            self.backend_instance.params.update(params)
            logging.info(f"self.backend_instance.params:{self.backend_instance.params}")

    def send_message_to_UI(self, element_id, innerHTML, ws_socket=None):

        msg = {"element_id": element_id, "innerHTML": innerHTML}
        msg = json.dumps(msg)

        if ws_socket:
            t_ = ws_socket.write_message(msg)
            asyncio.ensure_future(t_)

        else:  # broadcast
            for ws_ch in self.web_socket_channels:
                t_ = ws_ch.write_message(msg)
                asyncio.ensure_future(t_)

    def start_tornado(self):

        logging.info("starting tornado webserver on http://{}:{}...".format(LISTEN_ADDRESS, LISTEN_PORT))

        url_map = [
            (r"/", HttpHandler, {'parent': self}),
            (r'/websocket', WebsockHandler, {'parent': self}),
        ]

        self.tornado_instance = tornado.web.Application(url_map, **APPLICATION_OPTIONS)
        self.tornado_instance.listen(LISTEN_PORT, LISTEN_ADDRESS)
        tornado.platform.asyncio.AsyncIOMainLoop().install()

    def start_backend(self):

        logging.info(f"starting backend ...")

        self.backend_instance = Backend(parent=self)
        _t = self.backend_instance.run()
        asyncio.ensure_future(_t)

    def run(self):

        self.start_tornado()
        self.start_backend()

        asyncio.get_event_loop().run_forever()


def main():

    logging.basicConfig(
        stream=sys.stdout,
        level="INFO",
        format="[%(asctime)s]%(levelname)s %(funcName)s() %(filename)s:%(lineno)d %(message)s")

    a = Application()
    a.run()


if __name__ == '__main__':
    main()