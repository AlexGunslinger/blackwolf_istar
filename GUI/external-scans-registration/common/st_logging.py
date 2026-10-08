import streamlit as st
from streamlit.report_thread import REPORT_CONTEXT_ATTR_NAME
from threading import current_thread
from contextlib import contextmanager
from io import StringIO
import sys
import logging
import time


@contextmanager
def st_redirect(src, dst, placeholder, messages=[]):
    output_func = getattr(placeholder, dst)

    with StringIO() as buffer:
        old_write = src.write

        def new_write(b):
            messages.append(b)
            if getattr(current_thread(), REPORT_CONTEXT_ATTR_NAME, None):
                buffer.write(b + '')
                # all_msg = buffer.getvalue()
                # for msg in all_msg.split('\n'):
                #     output_func(msg)
                output_func(buffer.getvalue() + '')
            else:
                old_write(b)

        try:
            src.write = new_write
            yield
        finally:
            src.write = old_write


@contextmanager
def st_stdout(dst, placeholder, messages):
    "this will show the prints"
    with st_redirect(sys.stdout, dst, placeholder, messages):
        yield


@contextmanager
def st_stderr(dst, placeholder, messages):
    "This will show the logging"
    with st_redirect(sys.stderr, dst, placeholder, messages):
        yield


def demo_function():
    """
    Just a sample function to show how it works.
    :return:
    """
    for i in range(2):
        logging.info(f'Counting... {i}')
        time.sleep(0.1)
        print('Time out...\n')


if __name__ == '__main__':
    print_holder = st.empty()
    logging_holder = st.empty()
    tmp1 = []
    tmp2 = []
    with st_stdout("info", print_holder, tmp1) as a, st_stderr("code", logging_holder, tmp2) as b:
        demo_function()
        st.write('done')
        pass
    
    st.info(''.join(tmp1))
    st.spinner(''.join(tmp2))