class MKResults(object):
    def __init__(self, results_file):
        self.file = results_file

    def append_results(self, results):
        with open(self.file, mode='a') as fh:
            fh.write(results)