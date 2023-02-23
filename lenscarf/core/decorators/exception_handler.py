import traceback
import sys

def base(func):
    def inner_function(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as err:
            # expection formatter. Don't want all these logdecorator functions in the trace.
            _msg = "".join(traceback.format_exception(type(err), err, err.__traceback__))
            msg = ''
            skip = 0
            for line in _msg.splitlines():
                if skip > 0:
                    skip -=1
                else:
                    # Each decorator call comes with three lines of trace, and there are about 4 decorators for each exception..
                    if 'logdecorator' in line:
                        skip = 3
                    else:
                        msg += line + '\n'
            print(msg)
            sys.exit()
    return inner_function


# def base(func):
#     return func