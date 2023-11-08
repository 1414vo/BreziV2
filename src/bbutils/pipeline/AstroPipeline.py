import copy

class AstroPipeline():
    def __init__(self, steps):
        '''
        
        '''
        for step in steps:
            if type(step) != tuple or \
                not len(step) in [2, 4]:
                raise TypeError("Each step should be a tuple with 2 or 4 members.")
        self.steps = steps

    def fit(self, X, y):
        out = copy.deepcopy(X)
        for step in steps:
            if len(step) == 2:
                name, model = step
                train_on = None
                apply_on = None
            else:
                name, model, train_on, apply_on = step
            
            if type(out) == dict:
                if not train_on:
                    raise ValueError(f"Input was a dictionary but pipeline step {name} expected a list.")
                model.fit(out[train_on])
                
                for data_cat in apply_on:
                    out[data_cat] = model.transform(out[data_cat])
                    
    def predict(self, X):
        out = copy.deepcopy(X)
        for step in steps:
            if len(step) == 2:
                name, model = step
                train_on = None
                apply_on = None
            else:
                name, model, train_on, apply_on = step
            
            if type(out) == dict:
                if not train_on:
                    raise ValueError(f"Input was a dictionary but pipeline step {name} expected a list.")
                for data_cat in apply_on:
                    out[data_cat] = model.transform(out[data_cat])