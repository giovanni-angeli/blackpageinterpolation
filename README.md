# Black Page Interpolation

Scattered Data Interpolation in N-dim space based on Radia Basis Functions

With: 
  * a backend using asyncio, tornado, websocket server and 
  * a GUI running in Browser (HTML and minimal usage of basic JS)

=========

1. create a virtualenv 
>     $ export VIRTENV_ROOT=desired-virtenv_root-path
>     $ mkdir ${VIRTENV_ROOT}
>     $ virtualenv -p /usr/bin/python3 ${VIRTENV_ROOT}

2. clone this project in ${PROJECT_ROOT}
>     $ git clone git@github.com:giovanni-angeli/pyjamapeople.git

3. build Install in edit mode:
>     $ . ${VIRTENV_ROOT}/bin/activate
>     $ cd ${PROJECT_ROOT}               
>     $ pip install -e ./

4. Run:
>     $ blackpageinterpolation &
>     $ chromium http://127.0.0.1:8000/ &
>     $ firefox http://127.0.0.1:8000/ &

