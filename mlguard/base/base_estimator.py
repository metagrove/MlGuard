class BaseEstimator:

    def get_params(self):
        return self.__dict__

    def set_params(self, **params):

        for key, value in params.items():
            setattr(self, key, value)

        return self
    
"""
model.get_params()
model.set_params(alpha=0.1)
"""
