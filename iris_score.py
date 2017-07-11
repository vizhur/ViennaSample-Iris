def init():
    import numpy
    import scipy
    from sklearn.linear_model import LogisticRegression
    
    global model
    import pickle
    f = open('model.pkl', 'rb')
    model = pickle.load(f)
    f.close()

def run(inputString):
    import json
    import numpy
    try:
        input_list = json.loads(inputString)
    except ValueError:
        return "bad input: expecting a JSON encoded list of lists."
    input_array = numpy.array(input_list)
    if (input_array.shape != (1, 4)):
        return 'bad input: expecting a JSON encoded list of lists of shape (1,4).'
    return str(model.predict(input_array)[0])

if __name__ == '__main__':
    import json
    init()
    print (run(json.dumps([[1,2,3,1]])))
