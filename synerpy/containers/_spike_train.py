from ._base import BaseTimeSeries

# It is not clear that Spikes should inherit the Trial class.
# Maybe an event class would be best fit for this?
class Spikes(BaseTimeSeries):
    """
    A class representing a spike train, or a series of action potentials.
    """

    def __init__(self, file, data):
        super(Spikes, self).__init__(file = file, data = data)
        self.n_mus = len(self.data)
        self.time_range = (min(map(min, data)), max(map(max, data)))

    def __repr__(self):
        # return f'Spikes("{self.file}", time_range = {self.time_range})'
        return "Spikes" + super().__repr__() + ")"
       
    def segment(self, min_time, max_time):
        """
        Return a chunk of the dataset between time min_time and max_time
        """
        data = self.data.copy()
       
        # looping through each mu and getting only matching values within range
        out = list(map(lambda y: list(filter(lambda x: x > min_time and x < max_time, y)), data))
           
        # now making masked array, since every mu has different number of APs within range
        longest = max(map(len, out))
        for i, mu in enumerate(out):
           
            # add 0s for missing values
            dif = longest - len(mu)
            out[i] = out[i] + ([0] * dif)

        # coercing to masked array and returning as Spikes object
        out = np.ma.masked_values(out, 0)

        return Spikes(file = self.file, data = out)
