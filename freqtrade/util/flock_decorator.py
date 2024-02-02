import sys
import os

from fcntl import LOCK_EX
from fcntl import LOCK_NB
from fcntl import flock
from time import sleep

NO_BLOCK = 'nb'
BLOCK = 'block'
RETRY = 'retry'


class UnableToLock(Exception):
    pass

class InvalidMode(Exception):
    pass


def my_lock(lock_file, mode=NO_BLOCK, retries=1, timeout=1):
    """Use flock sys-call to protect an action from multiple concurrent calls

    This decorator will attempt to get an exclusive lock on the specified file
    by using the flock system call.  As long as all callers of a protected action
    use this decorator with the same lockfile, only 1 caller will be able to
    execute at a time, all others will fail, or will be blocked.

    Usage:

        @my_lock('/var/run/protected.lock')
        def protected():
            # do some potentialy unsafe actions

        # wait indefinetly for the lock
        @my_lock('/var/run/protected.lock', mdoe=BLOCK)
        def protected():
            # do some potentialy unsafe actions
        
        # If the initial lock failed retry 10 more times.
        @my_lock('/var/run/protected.lock', mode=RETRY, retries=300)
        def protected():
            # do some potentialy unsafe actions


    :param lock_file: full path to a file that will be used as a lock.
    :type lock_file: string
    :param mode: how should we run this? (BLOCK, NO_BLOCK, RETRY)
    :type mode: string.
    :param retries: If the initial lock failed, how many more times to retry.
    :type retries: int
    :param timeout: How long(seconds) should we wait before retrying the lock
    :type timeout: int
    """
    def decorator(target):

        def wrapper(*args, **kwargs):
            # touch the file to create it. (not necessarily needed.)
            # will raise IOError if permission denied.
            if not (os.path.exists(lock_file) and os.path.isfile(lock_file)):
                f = open(lock_file, 'a').close()

            operation = LOCK_EX
            if mode in [NO_BLOCK, RETRY]:
                operation = operation | LOCK_NB

            f = open(lock_file, 'a')
            if mode in [BLOCK, NO_BLOCK]:
                try:
                    flock(f, operation)
                except IOError:
                    raise UnableToLock('Unable to get exclusive lock.')

            elif mode == RETRY:
                for i in range(0, retries + 1):
                    try:
                        flock(f, operation)
                        break
                    except IOError:
                        if i == retries:
                            raise UnableToLock(f'Unable to get exclusive lock after {retries} retries x {timeout} seconds.')
                        sleep(timeout)

            else:
                raise InvalidMode('%s is not a valid mode.')

            # Execute the target
            result = target(*args, **kwargs)
            # Release the lock by closing the file
            f.close()
            return result
        return wrapper
    return decorator