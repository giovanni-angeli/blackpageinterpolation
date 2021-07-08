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
from functools import partial

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


class HttpHandler(tornado.web.RequestHandler):  # pylint: disable=too-few-public-methods, abstract-method

    parent = None

    def initialize(self, *args, **kwargs):

        self.parent = kwargs.pop('parent')
        super().initialize(*args, **kwargs)

    def get(self):

        ctx = {
            "title": "ColorInterpolator",
            "footer": "© Munchkin Music Co - https://www.zappa.com/",
            "params_panel": self.parent.backend_instance.render_params_html()
        }
        ret = self.render("index.html", **ctx)
        return ret


class WebsockHandler(tornado.websocket.WebSocketHandler):  # pylint: disable=abstract-method

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

        self.params_file_path = os.path.join(HERE, '..', '..', '_ignore_', 'params')

        self.pigment_combos = [
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
            rfb_name='linear',
            epsilon=1.0,
            n_of_rotate_sample=10,
            n_of_sites=1000,
            pigment_combo=self.pigment_combos[0],
            randomize_sample=1,
            n_of_closest_points=50)

        self.param_descriptions = {
            # ~ 'pigment_combo': f"{self.pigment_combos}",
        }

        self.model = None
        self.results = None

    def render_params_html(self):

        _params = self.params.copy()

        pigment_combo = _params.pop('pigment_combo')
        # ~ pigment_combo_options = [f'<option value="{n}">{i} {n}</option>' for i, n in enumerate(self.pigment_combos)]
        pigment_combo_options = []
        for i, n in enumerate(self.pigment_combos):
            if n == pigment_combo:
                item = f'<option value="{n}" selected>{i} {n}</option>'
            else:
                item = f'<option value="{n}">{i} {n}</option>'

            pigment_combo_options.append(item)

        html_ = ''

        html_ += '<table><tr>\n'

        html_ += '<td><table>\n'

        for k, v in _params.items():
            description = self.param_descriptions.get(k)
            html_ += f'<tr>\n'
            html_ += f'<td style="text-align:right;">{k}:</td>\n'
            html_ += f'<td style="text-align:left;"><input value="{v}" name="{k}" size="40" class="params_panel_item"></input></td>\n'
            html_ += f'<td style="">{description}</td>\n'
            html_ += f'</tr\n>'

        html_ += f'<tr>\n'
        html_ += f'<td style="text-align:right;width:30%;">pigment_combo:</td>\n'
        html_ += f'<td style="text-align:left;"><select name="pigment_combo" class="params_panel_item">{pigment_combo_options}</select></td>\n'
        html_ += f'<td style="width:10%;">{description}</td>\n'
        html_ += f'</tr>\n'

        html_ += '</table></td>\n'

        html_ += '<td><table>\n'

        list_ = os.listdir(self.params_file_path) if os.path.exists(self.params_file_path) else []
        options_ = "".join([f'<option value="{n}">{n}</option>' for n in sorted(list_)])

        html_ += f'<tr><td style="text-align:right;"><label>select a params file to load:</label></td></tr>\n'
        html_ += f'<tr><td style="text-align:right;"><select id="load_params_file_name">{options_}</select></td></tr>\n'

        html_ += f"""<tr><td style="text-align:right;"><input type="submit" id="load_params_btn" value="load_params" onclick="send_command('load_params');"/></td></tr>\n"""
        html_ += f'<tr><td style="text-align:right;"><label>insert a filename for storing params:</label></td></tr>\n'
        html_ += f"""<tr><td style="text-align:right;"><input type="text" id="store_params_file_name"/></td></tr>"""
        html_ += f"""<tr><td style="text-align:right;"><input type="submit" id="store_params_btn" value="store_params" onclick="send_command('store_params');"/></td></tr>\n"""

        html_ += '</table></td>\n'

        html_ += '</tr></table>\n'

        return html_

    def update_progress(self, ws_socket, progress):

        self.parent.send_message_to_UI(element_id='answer_display', innerHTML=progress, ws_socket=ws_socket)

    async def run_model(self, ws_socket, order_by='rgb', reverse=True):

        try:
            _params = self.params.copy()
            _params['epsilon'] = float(self.params['epsilon'])
            _params['n_of_rotate_sample'] = int(self.params['n_of_rotate_sample'])
            _params['n_of_sites'] = int(self.params['n_of_sites'])
            _params['randomize_sample'] = int(self.params['randomize_sample'])
            _params['mockup'] = False
            _params['n_of_closest_points'] = int(self.params['n_of_closest_points'])

            logging.info(f"_params:{_params}")

            self.model = Model(**_params)

            self.model.progress = partial(self.update_progress, ws_socket)

            self.results = await self.model.run()

        except Exception:  # pylint: disable=broad-except
            self.results = traceback.format_exc()
            logging.error(traceback.format_exc())

        # ~ _, results_html, _ = self.model.render_results_html(self.results)
        self.refresh_results(ws_socket, order_by=order_by, reverse=reverse)

    def store_results(self, ):

        data_dir = "/opt/PROJECTS/blackpageinterpolation/_ignore_/data/"

        self.results.sort(key=lambda x: x.get('error', 0) and x['error'])
        f_name = f"results_{self.params['n_of_sites']}_{self.params['n_of_rotate_sample']}_{self.params['pigment_combo']}.json"
        with open(os.path.join(data_dir, f_name), 'w') as f:
            json.dump(self.results, f, indent=2)

        return f_name

    def refresh_results(self, ws_socket, order_by='error', reverse=True):

        if order_by == 'R':
            self.results.sort(key=lambda x: x['target_rgb'][0], reverse=reverse)
        elif order_by == 'G':
            self.results.sort(key=lambda x: x['target_rgb'][1], reverse=reverse)
        elif order_by == 'B':
            self.results.sort(key=lambda x: x['target_rgb'][2], reverse=reverse)
        elif order_by == 'l':
            self.results.sort(key=lambda x: x['predicted_LabCh'][0], reverse=reverse)
        elif order_by == 'a':
            self.results.sort(key=lambda x: x['predicted_LabCh'][1], reverse=reverse)
        elif order_by == 'b':
            self.results.sort(key=lambda x: x['predicted_LabCh'][2], reverse=reverse)
        else:
            self.results.sort(key=lambda x: x.get(order_by) and x[order_by], reverse=reverse)

        _, html_body, _ = self.model.render_results_html(self.results)

        self.parent.send_message_to_UI(element_id='data_display', innerHTML=html_body, ws_socket=ws_socket)

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

    def store_params(self, f_name):

        logging.warning(f"f_name:{f_name}")

        if not os.path.exists(self.params_file_path):
            os.makedirs(self.params_file_path)

        with open(os.path.join(self.params_file_path, f_name), 'w') as f:
            json.dump(self.params, f, indent=2)

        html_ = self.render_params_html()
        self.parent.send_message_to_UI(element_id='params_panel', innerHTML=html_)

    def load_params(self, f_name):

        logging.warning(f"f_name:{f_name}")

        if not os.path.exists(self.params_file_path):
            os.makedirs(self.params_file_path)

        if os.path.exists(os.path.join(self.params_file_path, f_name)):
            with open(os.path.join(self.params_file_path, f_name)) as f:
                self.params = json.load(f)

        html_ = self.render_params_html()
        self.parent.send_message_to_UI(element_id='params_panel', innerHTML=html_)

    def update_params(self, params):

        self.params.update(params)

        self.params['epsilon'] = min(max(float(self.params['epsilon']), 0.0001), 1000.)
        self.params['n_of_rotate_sample'] = min(max(int(self.params['n_of_rotate_sample']), 0.0001), 1000.)
        self.params['n_of_sites'] = min(max(int(self.params['n_of_sites']), 2), 1000)
        self.params['randomize_sample'] = min(max(int(self.params['randomize_sample']), 0), 1)
        self.params['n_of_closest_points'] = min(max(int(self.params['n_of_closest_points']), 2), 1000)

        logging.info(f"self.params:{self.params}")


class Application:

    web_socket_channels = []
    tornado_instance = None
    backend_instance = None

    async def handle_message_from_UI(self, ws_socket, pack):

        try:

            index_ = ws_socket in self.web_socket_channels and self.web_socket_channels.index(ws_socket)
            pack = json.loads(pack)
            logging.info(f"index_:{index_}, pack:{pack}")

            command = pack.get('message', pack.get('command'))

            order_by = pack.get('option', {}).get('order_by', 'rgb')
            reverse = pack.get('option', {}).get('reverse')
            reverse = bool(reverse)

            if command == 'run_model':

                self.send_message_to_UI(
                    element_id='answer_display',
                    innerHTML='Running Model, please wait...',
                    ws_socket=ws_socket)
                await self.backend_instance.run_model(ws_socket, order_by=order_by, reverse=reverse)

            elif command == 'store_results':

                f_name = self.backend_instance.store_results()
                self.send_message_to_UI(
                    element_id='answer_display',
                    innerHTML=f'stored data to "{f_name}"',
                    ws_socket=ws_socket)

            elif command == 'order_by':

                self.backend_instance.refresh_results(ws_socket, order_by=order_by, reverse=reverse)

            elif command == 'stop_model':

                if self.backend_instance.model:
                    self.backend_instance.model.stop()

            elif command == 'params_panel':

                params = pack.get('option')
                self.backend_instance.update_params(params)

            elif command == 'store_params':

                f_name = pack.get('option', {}).get('f_name')
                if f_name:
                    self.backend_instance.store_params(f_name)
                    self.send_message_to_UI(
                        element_id='answer_display',
                        innerHTML=f'stored params to "{f_name}"',
                        ws_socket=ws_socket)

            elif command == 'load_params':

                f_name = pack.get('option', {}).get('f_name')
                if f_name:
                    self.backend_instance.load_params(f_name)
                    self.send_message_to_UI(
                        element_id='answer_display',
                        innerHTML=f'loaded params from "{f_name}"',
                        ws_socket=ws_socket)

        except Exception as e:  # pylint: disable=broad-except
            logging.error(traceback.format_exc())
            self.send_message_to_UI(element_id='answer_display', innerHTML=f'Exception: "{e}"', ws_socket=ws_socket)

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
