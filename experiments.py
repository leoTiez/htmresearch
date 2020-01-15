from itertools import product
import os


def main():
    # learningRates = [0.01, 0.1, 0.5]
    # forgettings = [0.0, 0.1, 0.2, 0.3]
    # outputCellCounts = [128, 512, 1024]
    # cellsCounts = [4, 8, 16]
    # sdrSizes = [5, 20, 40]
    # outputActivation = [0.5, 0.3, 0.1]
    # useSupport = [True, False]
    # useApicalTiebreak = [True, False]
    #
    # cart_prod = product(
    #     learningRates,
    #     forgettings,
    #     outputCellCounts,
    #     cellsCounts,
    #     sdrSizes,
    #     outputActivation,
    #     useSupport,
    #     useApicalTiebreak,
    # )
    #
    cwd = os.getcwd()
    # print "\n\nIncremental Bayesian"
    # for element in cart_prod:
    #     print element
    #     os.system("python %s/tests/bayesian/bayesian_convergence_activity_plot.py "
    #               "--implementation Bayesian "
    #               "--learningRate %s "
    #               "--forgetting %s "
    #               "--outputCount %s "
    #               "--cellCount %s "
    #               "--sdrSize %s "
    #               "--outputActivation %s "
    #               "--useSupport %s "
    #               "--useApicalTiebreak %s >> plot_test_incremental_exp.txt"
    #               % (cwd, element[0], element[1], element[2], element[3],
    #                  element[4], element[5], element[6], element[7])
    #               )

    # cart_prod = product(
    #     forgettings,
    #     outputCellCounts,
    #     cellsCounts,
    #     sdrSizes,
    #     outputActivation,
    #     useSupport,
    #     useApicalTiebreak,
    # )

    cart_prod = [
        [0.0, 1024, 8, 5, 0.3, True, True],
        [0.0, 512, 8, 20, 0.3, True, True],
        [0.0, 128, 8, 5, 0.3, True, True],
    ]
    
    print "\n\nSumming Bayesian"
    for element in cart_prod:
        os.system("python %s/tests/bayesian/bayesian_convergence_activity_plot.py "
                  "--implementation SummingBayesian "
                  "--forgetting %s "
                  "--outputCount %s "
                  "--cellCount %s "
                  "--sdrSize %s "
                  "--outputActivation %s "
                  "--useSupport %s "
                  "--useApicalTiebreak %s >> plot_test_summing_exp.txt"
                  % (cwd, element[0], element[1], element[2], element[3],
                     element[4], element[5], element[6])
                  )

    os.system("python %s/tests/bayesian/cp_gran_object_test.py "
              "--implementation Bayesian >> plot_test_numenta_exp.txt" % cwd)

if __name__ == '__main__':
    main()

