class OptimizationResult:
    def __init__(self,
                 x=None,
                 success=None,
                 status=None,
                 message=None,
                 fun=None,
                 jac=None,
                 hess=None,
                 nfev=None,
                 njev=None,
                 nhev=None,
                 nit=None,
                 tsec=None,
                 name=None,
                 nlp=None):
        localz = {k: v for k, v in locals().items() if k != "self"}
        self.__dict__.update(localz)

    def __repr__(self):
        r = ("OptimizationResult:\n"
             "  success: {}\n"
             "  fun:     {}\n"
             "  nit:     {}\n").format(self.success, self.fun,
                                       self.nit)
        if self.tsec:
            r += "  tsec:  {}\n".format(self.tsec)
        if self.status:
            r += "  status: {}\n".format(self.status)
        r += "  message: {}\n".format(self.message)
        return r

    def getValueList(self, keys):
        return [v for k, v in self.__dict__.items() if k in keys]
