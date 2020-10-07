class StatehandlingMeta(object):
    def add_callback(self,state,callback):
        if 'callbacks' in state:
            state['callbacks'].append(callback)
        else:
            state['callbacks'] = [callback]

    def handle_state(self,state,field_name,module,*params) -> tuple:
        if not self.align_with_next:
            return params + params
        else:
            a = state.get(field_name)
            if a is None:
                state[field_name] = {module: params}
                return (None,)*len(params)*2
            elif module in a:
                o_params = a[module]
                a[module] = params
            else:
                a[module] = params
                return (None,)*len(params)*2
            return o_params + params


def run_once(f):
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            return f(*args, **kwargs)
    wrapper.has_run = False
    return wrapper
