import cProfile
import pstats
from time import perf_counter as clock

class ProfileWrapper() :
    ''' A time profiler decorator. The function profile() is called
    at each loop and the data is stored. This allow to obtain a single
    profile output file with the average of the timings.
    '''
    def __init__(self   ):        
        self.pr = cProfile.Profile(self.clock_2)
        self.stats_objects = []
    
    def clock_2(self) :
        return 1000*clock()

    def profile(self , func):
        def wrapper(*args , **kwargs):
            self.pr.enable()
            res = func(*args, **kwargs)
            self.pr.disable()
            ps = pstats.Stats(self.pr)  
            self.stats_objects.append(ps)

            return res
        return wrapper    

    def print_stats(self, output_file ,  lines_to_print=None , sort_by='cumulative'):
        '''Args :
        - sort_by : sort type to sort the results
        - output_file (str) : name of the output file, 'cumulative'  , "name"
        - line_to_print : number of maximum line to print in the text file
        '''
        with open(output_file, 'w') as f:
            ps = pstats.Stats(self.pr, stream=f)
            

            if isinstance(sort_by, (tuple, list)):
                ps.sort_stats(*sort_by)
            else:
                ps.sort_stats(sort_by)
            ps.print_stats(lines_to_print)

        return 0