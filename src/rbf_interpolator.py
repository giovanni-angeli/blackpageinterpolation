# coding: utf-8

# pylint: disable=missing-docstring
# pylint: disable=logging-format-interpolation
# pylint: disable=line-too-long
# pylint: disable=invalid-name
# pylint: disable=broad-except
# pylint: disable=no-self-use
# pylint: disable=too-many-lines
# pylint: disable=too-many-statements
# pylint: disable=too-many-locals
# pylint: disable=too-many-nested-blocks


import os
import logging
import time
import json
import math
import random

import numpy as np  # pylint: disable=import-error

MOCKUP = False
DRY = 0
HERE = os.path.abspath(os.path.dirname(__file__))
RESULTS_PATH = os.path.join(HERE, 'results')


MAX_COL_VAL = 1000.0
MEASURE_SCALE_FACTOR = 1.


class RbfInterpolator:

    def __init__(self, r_b_function_name='gauss', sites=None, measures=None, epsilon=None):

        self.r_b_functions = {
            'test': self.test,
            'p_cubic': self.p_cubic,
            'gauss': self.gauss,
            'multiquadric': self.multiquadric,
            'inv_multiquadric': self.inv_multiquadric,
        }

        self.sites = sites
        self.measures = measures
        self.epsilon = epsilon
        self.r_b_function = self.r_b_functions[r_b_function_name]
        self.A = None
        self.C = None

    def compute_matrix(self):

        t0 = time.time()
        A_ = np.array([[self.r_b_function(x, y, self.epsilon) for x in self.sites] for y in self.sites])
        dt0 = time.time() - t0

        t0 = time.time()
        try:
            self.A = np.linalg.inv(A_)
        except Exception as e:
            logging.error(f"e:{e}")
        dt1 = time.time() - t0

        t0 = time.time()
        self.C = np.dot(self.measures, self.A)
        dt2 = time.time() - t0

        return (dt0, dt1, dt2, )

    def interpolate(self, x):

        fi = np.array([self.r_b_function(x, y, self.epsilon) for y in self.sites])
        # ~ print (f"fi:{fi}")
        # ~ logging.warning(f"self.C:{self.C}")
        r = np.dot(self.C.T, fi)
        r = max(0, r)
        return r

    @staticmethod
    def test(x, x_0, epsilon):

        d = x - x_0
        r = math.sqrt(np.dot(d.T, d))
        # ~ val = np.exp(- r * epsilon) ** 10
        # ~ val = max(0, (1 - r * epsilon))
        # ~ val = (1 - r * epsilon) ** 2
        val = max(0, (1 - r * epsilon))
        # ~ val = (1. - r * epsilon)
        # ~ val = (1 - r * 50. - r ** 2 * .5)
        # ~ val = math.sqrt(max(0, (1 - r * epsilon)))
        return val

    @staticmethod
    def p_cubic(x, x_0, epsilon):

        d = x - x_0
        r = epsilon * np.dot(d.T, d)
        if r > 0.001:
            r = r ** 4 * math.log(r)
        return r

    @staticmethod
    def multiquadric(x, x_0, epsilon):

        d = x - x_0
        r = np.sqrt(1 + 2 * epsilon * np.dot(d.T, d))
        return r

    @staticmethod
    def inv_multiquadric(x, x_0, epsilon):

        d = x - x_0
        r = 1. / np.sqrt(1 + 2. * epsilon * np.dot(d.T, d))
        return r

    @staticmethod
    def gauss(x, x_0, epsilon):

        d = x - x_0
        r = np.exp(- epsilon * np.dot(d.T, d))
        return r


class Model:

    def __init__(self, **kwargs):

        self.rfb_name = kwargs.get('rfb_name', 'test')
        self.epsilon = kwargs.get('epsilon', 1.0)
        self.n_of_rotate_sample = kwargs.get('n_of_rotate_sample', 100)
        self.n_of_sites = kwargs.get('n_of_sites', 10)
        self.mockup = kwargs.get('mockup', False)
        self.f_name = kwargs.get('f_name')
        self.offset = kwargs.get('offset', 0)

        self.__stop = False
        self.progress = None
        self.sites_to_data = None

    def stop(self):
        self.__stop = True
        logging.info(f"self.__stop:{self.__stop}")

    def create_mockup_data(self, N):

        def fun(s, i):
            c = np.array([.5, .5, .5])
            d = s - c
            r = np.exp(- 10. * i * np.dot(d.T, d))
            return r

        sites = [np.array([random.random(), random.random(), random.random()]) for i in range(N)]
        measures_dict = {}
        for i, k in enumerate(["white", "noir", "rouox", "jauox", "Neutro"]):
            measures_dict[k] = [fun(s, i) for s in sites]

        self.data = []

        return sites, measures_dict

    def load_data(self):

        # ~ logging.warning(f"self.f_name:{self.f_name}")

        with open(self.f_name) as f:
            data = json.load(f)

        filter_names = (
            "S 1030-Y10R",
            "S 3040-G60Y",
            "1030-Y10R",
            "S 3050-Y", 
            "S 4550-G", 
            "S 2040-Y10R",
            
            "S 5010-Y70R",
            "S 1502-Y50R",
            "S 4010-Y70R",
            "S 5010-G90Y",
            "S 2000-N",
            "S 1020-Y20R",
            "S 1015-Y90R",
            "S 4010-Y50R",
            "S 4020-Y",
            "S 4020-Y10R",
            "S 2002-Y",
            "S 3010-Y",
            "S 4005-Y20R",
            "S 1500-N",
            "S 4502-B",

            "S 5010-Y50R",
            "S 4050-Y70R",
            "S 4550-Y70R",

            "S 5010-Y30R",
            "S 4010-Y30R",
            "S 8502-Y",
            "S 3040-Y60R",
            "S 4040-Y70R",
            "S 6010-Y10R",
            "S 2502-Y",
            "S 8010-Y70R",
            "S 2040-Y20R",
            "S 3050-Y20R",
            "S 4040-Y10R",
            "S 5010-Y10R",
            "S 5040-Y70R",
            "S 6010-Y30R",
            "S 5030-Y",
            "S 2030-Y80R",
            )

        self.data = [d for d in data if d["StandardName"] not in filter_names][:self.n_of_sites]
        # ~ self.data = data[:self.n_of_sites]

        # ~ sites = [[i / MAX_COL_VAL for i in d["target_rgb"]] for d in data]
        # ~ sites = [(d["predicted_LabCh"][0] / 100., (d["predicted_LabCh"][1] + 1) / 2., (d["predicted_LabCh"][2] + 1) / 2.) for d in data]

        _offset = []
        _factor = []
        k = "predicted_LabCh"
        # ~ k = "target_rgb"
        for i in range(3):
            _min = min([d[k][i] for d in self.data])
            _max = max([d[k][i] for d in self.data])
            _offset.append(_min)
            _factor.append((_max - _min) * 2.0)

        # ~ _offset = [0, 0, 0]
        # ~ _factor = [255., 255., 255.]

        # ~ print(f"_offset:{_offset}, _factor:{_factor}")

        sites = []
        self.sites_to_data = []
        for d in self.data:
            _site = [(d["predicted_LabCh"][i] - _offset[i]) / _factor[i] for i in range(3)]
            sites.append(_site)

        measures_dict = {}

        pigment_names = [k for k in self.data[0]["ingredients"].keys() if k not in ('Master base', 'Neutro')]

        for k in pigment_names:
            measures_dict[k] = [MEASURE_SCALE_FACTOR * d["ingredients"][k] for d in self.data]

        return sites, measures_dict

    def run(self):

        np.set_printoptions(precision=3)

        if self.mockup:
            sites, measures_dict = self.create_mockup_data(self.n_of_sites)
        else:
            sites, measures_dict = self.load_data()

        self.n_of_rotate_sample = min(self.n_of_rotate_sample, len(sites))
        s = int((self.n_of_sites - self.n_of_rotate_sample) / 2)
        sample_range = range(s, s + self.n_of_rotate_sample)

        results = []
        progress = []
        for i in sample_range:

            error = 0
            result = {
                'name': self.data[i % len(sites)]['StandardName'],
                'predicted_LabCh': self.data[i % len(sites)]['predicted_LabCh'],
                'target_rgb': self.data[i % len(sites)]['target_rgb'],
            }
            for k, measures in measures_dict.items():
                # ~ logging.warning(f"k:{k}")

                meas = measures[:]
                test_values = np.array([meas.pop(i % len(sites)), ])
                measures_ = np.array(meas)

                s_ = sites[:]
                test_sites = np.array([s_.pop(i % len(sites))])
                sites_ = np.array(s_)

                interpolated_values = self.fit(sites_, measures_, test_sites)

                item = {
                    'measured': float(test_values[0]) / MEASURE_SCALE_FACTOR,
                    'interpolated': float(interpolated_values[0]) / MEASURE_SCALE_FACTOR,
                }

                error += np.sum(((test_values - interpolated_values) / MEASURE_SCALE_FACTOR) ** 2)
                result[k] = item
                result['rgb'] = [int(_ * MAX_COL_VAL) for _ in test_sites[0]]

                if self.__stop:
                    self.__stop = False
                    return []

            result['error'] = float(np.sqrt(error))
            print(f"error:{result['error']} N:{len(sites)}")
            results.append(result)
            if self.progress:
                progress = f"error:{result['error']:.2f} {len(results)}/{self.n_of_rotate_sample} N:{len(sites)}"
                self.progress(progress)

        results.sort(key=lambda x: x['error'])

        return results

    def filter_sites(self, sites, measures, test_sites):

        if 1:
            l_ = [[sites[i], measures[i]]  for i in range(len(sites))]
            # ~ print(f"l_:{l_}")

            def _dist(x):
                return np.linalg.norm(x[0] - test_sites[0])

            l_.sort(key=_dist)
            l_ = l_[:self.offset]

            _sites = np.array([i[0] for i in l_])
            _measures = np.array([i[1] for i in l_])

            # ~ print(f"_sites:{_sites}")

            return _sites, _measures

        return sites, measures

    def fit(self, sites, measures, test_sites):

        sites, measures = self.filter_sites(sites, measures, test_sites)

        interpolator = RbfInterpolator(self.rfb_name, sites, measures, self.epsilon)
        interpolator.compute_matrix()

        interpolated_values = [interpolator.interpolate(s) for s in test_sites]
        interpolated_values = np.array(interpolated_values)

        return interpolated_values

    @staticmethod
    def render_results_html(results, title):

        def _error_to_color(error):

            tmp = int(255 - error * 12)
            color = f'#FF{tmp:02X}{tmp:02X}'
            return color

        def _format_item_data(r, name, error):

            recipes = {'measured': {}, 'interpolated': {}, 'std_dev': error}
            for k, v in r.items():
                recipes['measured'][k] = f"{v['measured']:.4f}"
                recipes['interpolated'][k] = f"{v['interpolated']:.4f}"
            # ~ recipes = str(recipes).replace("'", ' ').replace('"', ' ').replace(':', ' ').replace('{', ' ').replace('}', ' ')
            recipes = f"""{name} {str(recipes).replace("'", ' ').replace('"', ' ')}"""

            html_ = ""

            html_ += f'<table title="{recipes}" style="table-layout:fixed;width:100%;" onclick="alert(\'{recipes}\');">'
            html_ += "<tr>"

            # ~ for k in r.keys():
            # ~ html_ += f"<th>{k}</th>"
            # ~ html_ += "<tr>"
            # ~ for v in r.values():
            # ~ diff = v['measured'] - v['interpolated']
            # ~ color = "#FFFFDD" if (abs(diff) > 1 or abs(diff) / max(0.001, v['measured']) > 0.5) else "#EEEEEE"
            # ~ html_ += f"""<td style="background-color:{color};">{v['measured']:8.3f}({diff:.3f})</td>"""

            for k, v in r.items():
                diff = v['measured'] - v['interpolated']
                color = "#FFFFDD" if abs(diff) / max(0.001, v['measured']) > 0.5 else "#EEEEEE"
                html_ += f"""<td style="background-color:{color};">{k}:<br/>{v['measured']:8.3f}({diff:.3f})</td>"""

            html_ += "</tr>"
            html_ += "</table>"

            return html_

        html_head = f"""<!DOCTYPE HTML>
        <html>
        <head>
        <meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
        <title> Color Interpolation Results </title>
        <style>table {{border: 1px solid #AAAAAA;font-family: monospace;}}</style>
        </head>
        """

        html_body = ""
        html_body += "<body>"
        html_body += """<table style="table-layout:fixed;width:100%;" cellspacing="2px" cellpadding="2px">"""
        html_body += """<tr><th>ord</th><th>name</th><th>chip</th><th>rgb</th><th width="65%">measured (measured - interpolated)</th><th>std dev</th></tr>"""
        for i, res in enumerate(results):
            r = res.copy()
            rgb = r.pop('rgb')
            target_rgb = r.pop('target_rgb')
            predicted_LabCh = r.pop('predicted_LabCh')
            name = r.pop('name')
            error = round(r.pop('error'), 2)
            rgb_hex = "#{:02X}{:02X}{:02X}".format(*[int(a) for a in target_rgb])
            html_body += f"""
            <tr>
                <td>{i:02d}</td>
                <td>{name}</td>
                <td style="border-radius:8px;padding:6px;background-color:{rgb_hex};width:200px;"></td>
                <td>{[int(a) for a in target_rgb]}</td>
                <td>{_format_item_data(r, name, error)}</td>
                <td style="background-color:{_error_to_color(error)};">{error}</td>
            </tr>
            """

        html_body += "</table>"
        html_body += "</body>"

        html_foot = """</html>"""

        return [html_head, html_body, html_foot]


def check():

    import glob

    for f_name in glob.glob("/opt/PROJECTS/color_match/data/recipes.Master base,Neutro,*.json"):

        m = Model(
            rfb_name='test',
            epsilon=1.,
            n_of_rotate_sample=100,
            n_of_sites=10,
            mockup=False,
            f_name=f_name,
            offset=0)

        m.load_data()


def run():

    for epsilon in (0.01, 0.1, 1.):

        if DRY:
            with open(os.path.join(RESULTS_PATH, f"{title}.json"), 'r') as f:
                results = json.load(f)
        else:
            m = Model(
                rfb_name='test',
                epsilon=epsilon,
                n_of_rotate_sample=100,
                n_of_sites=10,
                mockup=False,
                f_name="/opt/PROJECTS/color_match/data/recipes.Master base,Neutro,jauox,noir,rouox,white.json",
                offset=0)

            results = m.run()
            with open(os.path.join(RESULTS_PATH, f"{title}.json"), 'w') as f:
                json.dump(results, f, indent=2)

        with open(os.path.join(RESULTS_PATH, f"{title}.html"), 'w') as f:
            html_doc = "".join(Model.render_results_html(results, title))
            f.write(html_doc)


if __name__ == '__main__':
    check()
    # ~ run()
