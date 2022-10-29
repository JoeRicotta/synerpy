from ._base import BaseTimeSeries


class Rates(BaseTimeSeries):
    """
    A class built to handle instantaneous (i.e., continuous) firing rate data
    """
    
    time_units = "s"

    def __init__(self, file, data, time = None):
        super(Rates, self).__init__(file = file, data = data, time = time)
        self.n_mus = self.data.shape[1]

    @property
    def T(self):
        file, data, time = super().T
        return Rates(file, data, time)

    def __add__(self, other):
        data = super().__add__(other)
        return Rates(None, data = data, time = self.time)

    def __sub__(self, other):
        data = super().__sub__(other)
        return Rates(None, data = data, time = self.time)
    
    def __mul__(self, other):
        data = super().__mul__(other)
        return Rates(None, data = data, time = self.time)

    def __truediv__(self, other):
        data = super().__truediv__(other)
        return Rates(None, data = data, time = self.time)

    def __repr__(self):
        return "Rates" + super().__repr__() + ")"

    def append(self, other):
        """
        Joins two datasets rowwise, and tries to do so with time as well.
        """
        data, time = super().append(other)
        return Rates(None, data, time)

    def __or__(self, other):
        """
        Analogous to union. Joins the datasets columnwise, and returns common time-- or if uncommon, a None time.
        """        
        data, time = super().__or__(other)
        
        return Rates(None, data = data, time = time)

    def __xor__(self, other):
        """
        Joins motor units together across motor units, while accounting for
        possible aliasing between sets of motor units.
        The set of unique MUs as determined by a correlation less than .96 between all
        MUs; averages frequency data between two MUs when aliasing occurs.
        """
        
        thresh = .96
        added = self | other

        time = added.data.T[0].tolist()
        data = added.data.T[1:]

        # getting correlation matrix from joined data
        # and prepping data to be edited
        corr = np.corrcoef(data)
        
        to_del = []
        for i,a in enumerate(corr):
            for j,b in enumerate(a):
                if i <= j or i < self.n_mus or j < other.n_mus or b < thresh:
                    pass
                else:
                    # average the two motor unit data to comprise a new motor unit and remove the former
                    data[i] = (data[i] + data[j]) / 2
                    to_del.append(j)

        # deleting all indices
        to_del = list(set(to_del))
        to_del.sort(reverse = True)
        for ind in to_del:
            data = np.delete(data, ind, 0)
            
        data = data.tolist()
        data.insert(0, time)
        data = np.array(data).T

        # returning object
        return Rates(f"{self.file[0:4]}... & {other.file[0:4]}...", data = data, time = self.time)


    def segment(self, start_s, end_s):
       
        data, time = super().segment(start_s, end_s)

        return Rates(self.file, data, time)
   
   
    def downsample(self, n_chunks):
        """
        Downsample data to n_chunks in row length, along with time.
        """
        # calling super method
        data, time = super().downsample(n_chunks)

        return Rates(self.file, data, time)


