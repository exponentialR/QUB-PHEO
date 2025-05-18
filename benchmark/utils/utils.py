import os


class _MuteStderr:
    def __enter__(self):
        self.orig_stderr_fd = os.dup(2)
        os.close(2)
        os.open(os.devnull, os.O_RDWR)

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.close(2)
        os.dup2(self.orig_stderr_fd, 2)
        os.close(self.orig_stderr_fd)






