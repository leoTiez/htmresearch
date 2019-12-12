import numpy as np
from htmresearch.frameworks.layers.l2_l4_inference import L4L2Experiment


def create_object_single(
        num_objects,
        num_of_sensations,
        num_of_input,
        num_of_columns,
        noise=1
):
    object_dict = {}
    noise_object_dict = {}
    for o in range(num_objects):
        object_dict[o] = []
        noise_object_dict[o] = []
        for sensation in range(num_of_sensations):
            choice = np.arange(0, num_of_columns)
            sen1 = np.random.choice(choice, num_of_input, replace=False)
            sen2 = np.random.choice(choice, num_of_input, replace=False)
            sen1_noise = np.copy(sen1)
            sen1_noise[np.random.randint(0, num_of_input, noise)] = np.random.choice(choice, noise, replace=False)
            sen2_noise = np.copy(sen2)
            sen2_noise[np.random.randint(0, num_of_input, noise)] = np.random.choice(choice, noise, replace=False)

            sensation = (
                set(sen1),
                set(sen2)
            )
            sensation_noise = (
                set(sen1_noise),
                set(sen2_noise)
            )

            sensation_dict = {0: sensation}
            sensation_dict_noise = {0: sensation_noise}

            object_dict[o].append(sensation_dict)
            noise_object_dict[o].append(sensation_dict_noise)

    return object_dict, noise_object_dict


def printRecognition(object_rep, pred):
    for num, o in enumerate(object_rep):
        overlap = len(pred.intersection(o))
        print  "Object no %s: %s/%s of the neuron representation is active." % (num, overlap, len(o))

def test_summing_bayesian():
    # Init
    numCorticalColumnns = 1
    numLearningPoints = 3
    maxNumSegments = 5
    columns_count = 2048
    cells_per_column = 8

    # Create objects
    num_objects = 10
    num_of_sensations = 5
    num_of_input = 15

    # Recognition
    rec_object = 1
    repetition = 3

    L2Overrides = {
        "noise": 1e-8,
        "cellCount": 512,  # new: 256 # original: 4096
        "inputWidth": cells_per_column * columns_count,  # new: 8192 # original: 16384 (?) = cells per column * column count
        "sdrSize": 40
    }

    L4Overrides = {
        "noise": 1e-8,
        "cellsPerColumn": cells_per_column,  # new: 4 # original 32
        "columnCount": columns_count,  # new: 2048 # original: 2048
        "minThreshold": 0.1,
    }

    exp1 = L4L2Experiment(
        'single_column',
        implementation='SummingBayesian',
        L2RegionType="py.BayesianColumnPoolerRegion",
        L4RegionType="py.BayesianApicalTMPairRegion",
        L2Overrides=L2Overrides,
        L4Overrides=L4Overrides,
        numCorticalColumns=numCorticalColumnns,
        maxSegmentsPerCell=maxNumSegments,
        numLearningPoints=numLearningPoints,  # number repetitions for learning
        seed=1
    )

    objects, noise_objects = create_object_single(num_objects, num_of_sensations, num_of_input, 1024) # TODO Why 1024

    # learn the sensations
    print "Train objects"
    exp1.learnObjects(objects)
    exp1.sendReset()

    object_representations = []
    for o in range(num_objects):
        rep = exp1.objectL2Representations[o][0]
        object_representations.append(rep)

    for _ in range(repetition):
        np.random.shuffle(noise_objects[rec_object])
        for num, sensation in enumerate(noise_objects[rec_object]):
            exp1.infer([sensation], objectName=rec_object, reset=False)

            object_prediction = exp1.getL2Prediction()[0]

            print "\nStep: %s" % num
            printRecognition(object_representations, object_prediction)


if __name__ == '__main__':
    test_summing_bayesian()

