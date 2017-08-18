if 'axes.formatter.useoffset' in matplotlib.rcParams:
        # None triggers use of the rcParams value
            useoffsetdefault = None
        else:
                # None would raise an exception
                    useoffsetdefault = True

                    class FixedScalarFormatter(matplotlib.ticker.ScalarFormatter):
                            def __init__(self, format, useOffset=useoffsetdefault, useMathText=None, useLocale=None):
                                        super(FixedScalarFormatter,self).__init__(useOffset=useOffset,useMathText=useMathText,useLocale=useLocale)
                                                self.base_format = format
                                                    def _set_format(self, vmin, vmax):
                                                                """ Calculates the most appropriate format string for the range (vmin, vmax).

                                                                        We're actually just using a fixed format string.
                                                                                """
                                                                                        self.format = self.base_format
                                                                                                if self._usetex:
                                                                                                                self.format = '$%s$' % self.format
                                                                                                                        elif self._useMathText:
                                                                                                                                        self.format = '$\mathdefault{%s}$' % self.format   
