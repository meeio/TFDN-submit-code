
class EndlessIter:

    """ A helper class which iter a dataloader endnessly
    
    Arguments:
        dataloader {DataLoader} -- Dataloader want to iter
    
    """

    def __init__(self, dataloader, max = -1):
        assert dataloader is not None
        self.l = dataloader
        self.it = None
        self.current_iter = 0
        self.max_iter = max

    def next(self, need_end=False):
        """ return next item of dataloader, if use 'endness' mode, 
        the iteration will not stop after one epoch
        
        Keyword Arguments:
            need_end {bool} -- weather need to stop after one epoch
             (default: {False})
        
        Returns:
            list -- data
        """
        self.current_iter += 1
        if self.max_iter > 0 and self.current_iter > self.max_iter:
            self.current_iter = 0
            return None

        if self.it == None:
            self.it = iter(self.l)

        try:
            i = next(self.it)
        except Exception:
            self.it = iter(self.l)
            self.current_iter = 0
            i = next(self.it) if not need_end else None

        return i

