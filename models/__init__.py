from .M3DFEL import M3DFEL

def create_model(args):
    
    model = M3DFEL(args)
    
    return model