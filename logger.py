'''
For Logging

Can redirect standard out and standard error to a file

@author austin
'''

import os
from datetime import datetime
import sys

version = '1.0.0'

change_history = '''
date       author    version    message
###############################################################
09/25/19   austin    1.0.0      Initial version
###############################################################
'''

def about():
    print('-------------------------------------------------')
    print('Name:', os.path.basename(__file__))
    print('            Change History',change_history)
    print('Current Version:', version)
    print('-------------------------------------------------')

'''
if 'DEBUG' in os.environ:
    DEBUG = int(os.environ['DEBUG'])
else:
    DEBUG = 0
'''

class Logger():
    '''Logging object, allows you to redirect logs to file'''
    def __init__(self, debug=0, log_file=None, error_file=None):
        self.debug = debug
        if log_file is not None:
            self.original_stdout = sys.stdout
            try:
                self.redirect_stdout(log_file)
            except Exception as e:
                self.reset_stdout()
                print('Unable to redirect stdout', e)
        if error_file is not None:
            self.original_stderr = sys.stderr
            try:
                self.redirect_stderr(error_file)
            except Exception as e:
                self.reset_stderr()
                print('Unable to redirect stderr', e)

    def redirect_stdout(self, file):
        sys.stdout = open(file, 'a+')

    def reset_stdout(self):
        sys.stdout = self.original_stdout

    def redirect_stderr(self, file):
        sys.stderr = open(file, 'a+')

    def reset_stderr(self):
        sys.stderr = self.original_stderr




    def put_msg(self, msg_type, msg, name=' ', line=''):
        '''Message type can be:
                I for informational
                D for debug only (only prints if debug is set to 1)
                E for error
            Timestamp is format YYYY-MM-DD_HH:MM:SS
            '''
        if self.debug == 0 and msg_type == 'D':
            return

        time = datetime.now().strftime(r'%Y-%m-%d_%H:%M:%S')
        output = f'{msg_type}:{time}'

        if line != ' ':
            line = str(line) # Safety
            # Pad the line number
            while len(line) < 4:
                line = ' ' + line
            output = output + f' @ {line}'
        if name != ' ':
            name = str(name) # Safety
            output = output + f' from {name}'

        output = output + ' - ' + str(msg)

        print(output)



if __name__ == '__main__':
    logger = logger.Logger(debug=1)
    logger.put_msg('D', 'Debug message', name=__file__, line=68)
    logger.put_msg('I', 'Information message', name=__file__, line=69)
    logger.put_msg('E', 'Error message', name=__file__, line=70)
    exit()