import os
import logging
import time
import sys

log = None

def setup_logging(clean=False, debug=False):
    global log
    
    if log is not None:
        return log
    
    try:
        if clean and os.path.isfile('setup.log'):
            os.remove('setup.log')
        time.sleep(0.1) # prevent race condition
    except:
        pass
    
    if sys.version_info >= (3, 9):
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s | %(levelname)s | %(pathname)s | %(message)s', filename='setup.log', filemode='a', encoding='utf-8', force=True)
    else:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s | %(levelname)s | %(pathname)s | %(message)s', filename='setup.log', filemode='a', force=True)

    log = logging.getLogger("sd")

    return log
