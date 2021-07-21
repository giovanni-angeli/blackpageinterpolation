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

# pylint: disable=unused-variable


import os
import logging
import time
import json
import math
import random
import asyncio
import inspect
import glob
import csv

import numpy as np  # pylint: disable=import-error

import colormath.color_conversions  # pylint: disable=import-error

DRY = 0
HERE = os.path.abspath(os.path.dirname(__file__))
RESULTS_PATH = os.path.join(HERE, 'results')


def is_inside(p, cloud):

    t0 = time.time()
    cntrs = [0, 0]
    for q in cloud:
        p_q = (p - q)
        p_q_2 = np.dot(p_q, p_q)
        for x in cloud:
            if x is not q:
                d = np.dot((x - q), p_q) - p_q_2
                if d >= 0:
                    cntrs[0] += 1
                else:
                    cntrs[1] += 1
    dt = time.time() - t0
    r = abs(cntrs[0] - cntrs[1]) / (cntrs[0] + cntrs[1])
    # ~ print (f"p:{p}, r:{r:.3f}, cntrs:{cntrs}, dt:{dt}")
    return r


class RbfInterpolator:  # pylint: disable=too-many-instance-attributes

    def __init__(self, r_b_function_name='gauss', sites=None, measures=None, epsilon=None):

        # ~ method_list = inspect.getmembers(self, predicate=inspect.isfunction)
        # ~ logging.warning(f"method_list : {[a for a, b in method_list]}")
        # ~ logging.warning(f"method_dict : {dict(method_list)}")

        self.r_b_gradients = {
            # ~ 'test': self.gradient_test,
            'linear': self.linear_gradient,
            'exp': self.exp_gradient,
        }

        self.r_b_functions = dict(inspect.getmembers(self, predicate=inspect.isfunction))

        # ~ {
        # ~ 'test': self.test,
        # ~ 'linear': self.linear,
        # ~ 'p_cubic': self.p_cubic,
        # ~ 'gauss': self.gauss,
        # ~ 'multiquadric': self.multiquadric,
        # ~ 'inv_multiquadric': self.inv_multiquadric,
        # ~ }

        self.sites = sites
        self.measures = measures
        self.epsilon = epsilon
        self.r_b_function = self.r_b_functions[r_b_function_name]
        self.r_b_gradient = self.r_b_gradients.get(r_b_function_name)
        self.A = None
        self.lambdas = None

    def compute_lambdas(self):

        t0 = time.time()
        A_ = np.array([[self.r_b_function(x, y, self.epsilon) for x in self.sites] for y in self.sites])
        dt0 = time.time() - t0

        t0 = time.time()
        self.A = np.linalg.inv(A_)

        dt1 = time.time() - t0

        t0 = time.time()
        self.lambdas = np.dot(self.measures, self.A)
        dt2 = time.time() - t0

        # ~ logging.warning(f"self.A:{self.A}")
        # ~ logging.warning(f"self.lambdas:{self.lambdas}")

        return (dt0, dt1, dt2, )

    def gradient(self, x):

        ret = np.zeros(x.shape)
        if self.r_b_gradient:
            ret = self.r_b_gradient(x)
        return ret

    def interpolate(self, x):

        fi = np.array([self.r_b_function(x, x_k, self.epsilon) for x_k in self.sites])
        y = np.dot(self.lambdas.T, fi)
        y = max(0, y)
        return y

    @staticmethod
    def test(x, x_0, epsilon):

        d = x - x_0
        r = np.linalg.norm(d)
        val = np.exp(- epsilon * r)
        val = val ** 4
        return val

    @staticmethod
    def exp(x, x_0, epsilon):

        d = x - x_0
        r = np.linalg.norm(d)
        val = np.exp(- epsilon * r)
        return val

    @staticmethod
    def linear(x, x_0, epsilon):

        d = x - x_0
        r = np.linalg.norm(d)
        val = 1 - r * epsilon
        val = max(0, val)
        return val

    @staticmethod
    def p_cubic(x, x_0, epsilon):

        d = x - x_0
        r = epsilon * np.dot(d.T, d)
        if r > 0.001:
            r = r ** 4 * math.log(1 + r)
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
        r = np.dot(d.T, d)
        val = np.exp(- epsilon * r)
        return val

    def linear_gradient(self, x):

        diffs = np.array([(x - x_k) for x_k in self.sites])
        norm_diffs = diffs / np.linalg.norm(diffs)
        g = - .5 * self.epsilon * np.dot(self.lambdas.T, norm_diffs)
        return g

    def exp_gradient(self, x):

        diffs = np.array([(x - x_k) for x_k in self.sites])
        norm_diffs = diffs / np.linalg.norm(diffs)
        tmp_ = np.exp(- 1 * self.epsilon * diffs) * norm_diffs
        g = - .5 * self.epsilon * np.dot(self.lambdas.T, tmp_)
        return g


class Model:  # pylint: disable=too-many-instance-attributes

    def __init__(self, **kwargs):

        self.__params = kwargs

        self.json_data_dir = "/opt/PROJECTS/blackpageinterpolation/_ignore_/data/recipes"
        self.csv_data_dir = "/opt/PROJECTS/blackpageinterpolation/_ignore_/data/csv"

        # ~ self.f_name = "/opt/PROJECTS/blackpageinterpolation/_ignore_/data/Combinazioni_per_spazio_colore_interpolare.csv"

        self.rfb_name = kwargs.get('rfb_name', 'test')
        self.epsilon = kwargs.get('epsilon', 1.0)
        self.n_of_rotate_sample = kwargs.get('n_of_rotate_sample', 100)
        self.n_of_sites = kwargs.get('n_of_sites', 10)
        self.mockup = kwargs.get('mockup', False)
        self.n_of_closest_points = kwargs.get('n_of_closest_points', 0)
        self.randomize_sample = kwargs.get('randomize_sample', 0)
        self.pigment_combo = kwargs.get('pigment_combo')
        self.__stop = False
        self.progress = lambda x: x
        self.global_err = 0
        self.global_delta_E = 0

        self.data = None
        self.interpolator = None

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

    def load_data(self, ):

        if 'csv' in self.pigment_combo:
            sites, measures_dict = self.__load_data_csv()
        else:
            sites, measures_dict = self.__load_data_json()

        return sites, measures_dict

    def __load_data_csv(self, ):

        f_name = os.path.join(self.csv_data_dir, self.pigment_combo)

        filter_ids = (
            '',
            # ~ 85,
            # ~ 74,
            # ~ 266,
            # ~ 267,

            # ~ 7,
            # ~ 2,
            # ~ 51,
            # ~ 54,
            # ~ 56,
            # ~ 57,
            # ~ 58,
            # ~ 66,
            # ~ 67,
            # ~ 71,
            # ~ 81,
            # ~ 83,
        )

        measures_dict = {}
        sites = []
        self.data = []

        with open(f_name) as csvfile:
            reader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
            for row in reader:
                # ~ if int(row['ID']) not in filter_ids and float(row.get("Noir").replace(",", ".")) < 1.0001:
                # ~ if int(row['ID']) not in filter_ids and float(row.get("TiO2").replace(",", ".")) > 0.101:
                # ~ if int(row['ID']) not in filter_ids and abs(float(row.get('a*mis').replace(",", "."))) > 0.2:
                if int(row['ID']) not in filter_ids:
                    try:
                        lab_mis = [float(row.get(k)) for k in ('L*mis', 'a*mis', 'b*mis')]
                    except:
                        logging.warning(f"skipping row:{row}")
                        continue

                    # ~ logging.warning(f"row:{row}")
                    sites.append(lab_mis)

                    for k in ('TiO2', 'Noir', 'Rougeox', 'Jauox'):
                        measures_dict.setdefault(k, [])
                        measures_dict[k].append(float(row.get(k).replace(",", ".")))

                    c_lab = colormath.color_objects.LabColor(*lab_mis)
                    c_rgb = colormath.color_conversions.convert_color(c_lab, colormath.color_objects.sRGBColor)

                    row['target_rgb'] = c_rgb.get_upscaled_value_tuple()
                    row['StandardName'] = row['ID']
                    row['lab_mis'] = lab_mis
                    # ~ row['lab_mis'] = c_rgb.get_upscaled_value_tuple()
                    self.data.append(row)

        _offset = [0, 0, 0]
        _factor = [0, 0, 0]
        for i in range(3):
            _min = min([s[i] for s in sites])
            _max = max([s[i] for s in sites])
            _offset[i] = _min
            _factor[i] = (_max - _min) * 10.

        # ~ logging.warning(f"_offset:{_offset}, _factor:{_factor}")
        # ~ logging.warning(f"self.data:{self.data}")

        sites_ = []
        for s in sites:
            _site = [(s[i] - _offset[i]) / _factor[i] for i in range(3)]
            sites_.append(_site)

        # ~ sites_ = [[], [], []]
        # ~ for i in range(3):
            # ~ sites_[i] = [(s - _offset[i]) / _factor[i] for s in sites[i]]

        for i in range(3):
            _min = min([s[i] for s in sites_])
            _max = max([s[i] for s in sites_])
            # ~ logging.warning(f"i:{i}, _min:{_min}, _max:{_max}, ")

        return sites_, measures_dict

    def __load_data_json(self, data_format_to_use="predicted_LabCh"):  # predicted_LabCh, target_rgb

        f_name = os.path.join(
            self.json_data_dir,
            "recipes.Master base,Neutro," +
            self.pigment_combo +
            ",white.json")

        # ~ logging.warning(f"f_name:{f_name}")

        with open(f_name) as f:
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

        # ~ self.data = [d for d in data if d["StandardName"] not in filter_names][:self.n_of_sites]
        self.data = data[:self.n_of_sites]

        _offset = []
        _factor = []
        for i in range(3):
            _min = min([d[data_format_to_use][i] for d in self.data])
            _max = max([d[data_format_to_use][i] for d in self.data])
            _offset.append(_min)
            _factor.append((_max - _min) * 2.0)

        # ~ logging.warning(f"_offset:{_offset}, _factor:{_factor}")
        # ~ logging.warning(f"self.data:{self.data}")

        sites = []
        for d in self.data:
            _site = [(d[data_format_to_use][i] - _offset[i]) / _factor[i] for i in range(3)]
            sites.append(_site)
            d['lab_mis'] = d[data_format_to_use]

        pigment_names = [k for k in self.data[0]["ingredients"].keys() if k not in ('Master base', 'Neutro')]
        measures_dict = {}
        for k in pigment_names:
            measures_dict[k] = [d["ingredients"][k] for d in self.data]

        # ~ logging.warning(f"sites:{sites}, measures_dict:{measures_dict}")
        # ~ logging.warning(f"sites:{sites}")

        return sites, measures_dict

    def filter_sites(self, sites, measures, test_sites):

        l_ = list(zip(sites, measures))

        def _dist(x):
            return np.linalg.norm(x[0] - test_sites[0])
        l_.sort(key=_dist)
        l_ = l_[:self.n_of_closest_points]
        _sites = np.array([x[0] for x in l_])
        _measures = np.array([x[1] for x in l_])

        return _sites, _measures

    async def run(self):

        np.set_printoptions(precision=3)

        if self.mockup:
            sites, measures_dict = self.create_mockup_data(self.n_of_sites)
        else:
            sites, measures_dict = self.load_data()

        self.n_of_rotate_sample = min(self.n_of_rotate_sample, len(sites))
        if self.randomize_sample:
            sample_range = random.sample(range(0, len(sites)), self.n_of_rotate_sample)
        else:
            s = int((self.n_of_sites - self.n_of_rotate_sample) / 2)
            sample_range = range(s, s + self.n_of_rotate_sample)

        results = []
        for i in sample_range:

            i = i % len(sites)

            result = {
                'name': self.data[i]['StandardName'],
                'lab_mis': self.data[i]['lab_mis'],
                'target_rgb': self.data[i]['target_rgb'],
                'opacity': float(self.data[i]["Opacitàmis (%)"]),
            }

            for k, measures in measures_dict.items():

                meas = measures[:]
                test_values = np.array([meas.pop(i), ])
                measures_ = np.array(meas)

                s_ = sites[:]
                test_sites = np.array([s_.pop(i)])
                sites_ = np.array(s_)

                diffs_ = np.matrix([(test_sites[0] - x[0]) for x in sites_])

                sites_, measures_ = self.filter_sites(sites_, measures_, test_sites)

                # ~ diffs_ = np.matrix([(test_sites[0] - x[0]) for x in sites_])

                self.interpolator = RbfInterpolator(self.rfb_name, sites_, measures_, self.epsilon)
                self.interpolator.compute_lambdas()

                interpolated_values = [self.interpolator.interpolate(s) for s in test_sites]
                interpolated_values = np.array(interpolated_values)

                item = {
                    'measured': float(test_values[0]),
                    'interpolated': float(interpolated_values[0]),
                    'gradient': self.interpolator.gradient(test_sites[0])
                }
                result[k] = item
                await asyncio.sleep(0.0001)
                if self.__stop:
                    self.__stop = False
                    return []

            N = len(diffs_)
            sum_of_diffs = np.sum(diffs_)
            norm_diffs = np.array([x / np.linalg.norm(x) for x in diffs_])

            # ~ logging.warning(f"diffs_:{diffs_}, sum_of_diffs:{sum_of_diffs}")

            is_inside_ = is_inside(test_sites[0], sites_)
            dist_0 = np.linalg.norm(np.sum(norm_diffs)) / len(norm_diffs)
            # ~ result['dists'] = [is_inside_, dist_0, ]
            # ~ result['dists'] = [is_inside_, ]

            dist_0 = np.linalg.norm(sum_of_diffs) / N
            dist_1 = sum([np.linalg.norm(x) for x in diffs_]) / N
            dist_2 = min([np.linalg.norm(x) for x in diffs_])
            result['dists'] = [dist_0, dist_1, dist_2]

            error = sum([abs((result[k]['measured'] - result[k]['interpolated']) / max(0.1, result[k]['measured']))
                         for k in measures_dict.keys()])
            result['error'] = float(np.sqrt(error))

            jacobian = np.matrix([v['gradient'] for v in result.values() if isinstance(v, dict)])
            jacobian_inv = np.linalg.pinv(jacobian)
            dx = np.array([(result[k]['measured'] - result[k]['interpolated']) for k in measures_dict.keys()])
            dy = np.dot(jacobian_inv, dx)
            # ~ logging.warning(f'result["name"]:{result["name"]}, dx:{dx}, dy({dy.shape}):{dy}')

            result['delta_E'] = float(np.linalg.norm(dy))

            results.append(result)

            prog_mesg = f"delta_E:{result['delta_E']:.2f} error:{result['error']:.2f} {len(results)}/{self.n_of_rotate_sample} N:{len(sites)}"
            if self.progress and callable(self.progress):
                self.progress(prog_mesg)

        self.global_err = sum([r['error'] for r in results])
        self.global_delta_E = sum([r['delta_E'] for r in results])
        prog_mesg = f"global_delta_E:{self.global_delta_E:.2f}({self.global_delta_E/len(results):.2f}), "
        prog_mesg += f"global_err:{self.global_err:.2f}({self.global_err/len(results):.2f}), "
        prog_mesg += f"{len(results)}/{self.n_of_rotate_sample} N:{len(sites)}"
        prog_mesg += f" <br/> <small>{self.__params}</small>"
        logging.warning(prog_mesg)
        if self.progress and callable(self.progress):
            self.progress(prog_mesg)

        results.sort(key=lambda x: x['error'], reverse=True)

        for r in results:
            for k in r.keys():
                if isinstance(r[k], dict):
                    if r[k].get("gradient") is not None:
                        # ~ r[k]["gradient"] = float(np.sum(np.abs(r[k]["gradient"])))
                        r[k]["gradient"] = [float(i) for i in r[k]["gradient"]]

        # ~ logging.warning(f"results:{results}")

        return results

    @staticmethod
    def render_results_html(results):

        def _error_to_color(error):

            tmp = min(255, int(255 - error * 12))
            color = f'#FF{tmp:02X}{tmp:02X}'
            return color

        def _dist_to_color(dist):

            tmp = min(255, int(255 - dist * 120))
            color = f'#{tmp:02X}{tmp:02X}FF'
            return color

        def _format_item_data(r, name, error):

            recipes = {'measured': {}, 'interpolated': {}, 'std_dev': error}
            for k, v in r.items():
                recipes['measured'][k] = f"{v['measured']:.4f}"
                recipes['interpolated'][k] = f"{v['interpolated']:.4f}"
            recipes = f"""{name} {str(recipes).replace("'", ' ').replace('"', ' ')}"""

            html_ = ""

            html_ += f'<table title="{recipes}" style="table-layout:fixed;width:100%;" onclick="alert(\'{recipes}\');">'
            html_ += "<tr>"

            for k, v in r.items():
                diff = v['measured'] - v['interpolated']
                color = "#FFFFDD" if abs(diff) / max(0.1, v['measured']) > 0.5 else "#EEEEEE"
                html_ += f"""<td style="background-color:{color};">{k}:<br/>{v['measured']:.3f} {v['interpolated']:.3f}({diff:.3f})</td>"""

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
        html_body += """<tr>
            <th width="2%">ord</th><th>name</th>
            <th>chip</th>
            <th width="12%">rgb lab</th>
            <th width="60%">measured interpolated (diff)</th>
            <th>opacity</th>
            <th>std dev</th>
            <th>dists</th>
        </tr>"""

        for i, res in enumerate(results):
            r = res.copy()
            # ~ rgb = r.pop('rgb')
            target_rgb = r.pop('target_rgb')
            _rgb_string = " ".join([str(int(a)) for a in target_rgb])
            lab_mis = r.pop('lab_mis')
            _lab_string = ",".join([f"{a:.2f}" for a in lab_mis[:3]])
            name = r.pop('name')
            error = round(r.pop('error'), 2)
            dists = [round(d, 3) for d in r.pop('dists')]
            delta_E = round(r.pop('delta_E'), 2)
            opacity = round(r.pop('opacity'), 1)
            _color_dists = _dist_to_color(dists[-1])
            _color_delta_E = _error_to_color(delta_E * .5)
            rgb_hex = "#{:02X}{:02X}{:02X}".format(*[int(a) for a in target_rgb])
            html_body += f"""
            <tr>
                <td>{i:02d}</td>
                <td>{name}</td>
                <td style="border-radius:8px;padding:6px;background-color:{rgb_hex};width:200px;"></td>
                <td>{_rgb_string}<br/>{_lab_string}</td>
                <td>{_format_item_data(r, name, error)}</td>
                <td>{opacity}</td>
                <td style="background-color:{_error_to_color(error)};">{error}</td>
                <td style="background-color:{_color_dists};">{dists}</td>
            </tr>
            """

        html_body += "</table>"
        html_body += "</body>"

        html_foot = """</html>"""

        return [html_head, html_body, html_foot]


def check():

    for f_name in glob.glob("/opt/PROJECTS/color_match/data/recipes.Master base,Neutro,*.json"):

        m = Model(
            rfb_name='test',
            epsilon=1.,
            n_of_rotate_sample=100,
            n_of_sites=10,
            mockup=False,
            f_name=f_name,
            n_of_closest_points=0)

        m.load_data()


def run():

    for epsilon in (0.01, 0.1, 1.):

        m = Model(
            rfb_name='test',
            epsilon=epsilon,
            n_of_rotate_sample=100,
            n_of_sites=10,
            mockup=False,
            f_name="/opt/PROJECTS/color_match/data/recipes.Master base,Neutro,jauox,noir,rouox,white.json",
            n_of_closest_points=0)

        results = m.run()
        with open(os.path.join(RESULTS_PATH, "results.Master base,Neutro,jauox,noir,rouox,white.json"), 'w') as f:
            json.dump(results, f, indent=2)

        with open(os.path.join(RESULTS_PATH, "results.Master base,Neutro,jauox,noir,rouox,white.html"), 'w') as f:
            html_doc = "".join(Model.render_results_html(results))
            f.write(html_doc)


if __name__ == '__main__':
    check()
    # ~ run()
