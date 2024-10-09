# ============================================================================
# Record linkage software for the Data Wrangling course, 2021.
# Version 1.0
#
# =============================================================================

"""Main module for linking records from two files.

   This module calls the necessary modules to perform the functionalities of
   the record linkage process.
"""
# =============================================================================
# Import necessary modules (Python standard modules first, then other modules)

import time

from meta_tl.attribute_matching.string_instance_matching import QGramInstanceMatching
from meta_tl.feature_discovery import feature_distribution_analysis, classification_result_analysis
from meta_tl.feature_discovery.selection.top_selector import TopSelector

from classification.machine_learning.active_learning_solution import ActiveLearningBootstrap
from record_linkage import loadDataset
from record_linkage.blocking import blocking_functions_solution as blocking_functions, blocking as blocking
from record_linkage.classification.machine_learning import constants, active_learning_solution
from record_linkage.comparison import string_functions_solution as string_functions, comparison
from record_linkage.classification import threshold_classification_solution as threshold_classification
from record_linkage.evaluation import evaluation_solution as evaluation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# =============================================================================
from comparison.qgram_converter import QgramConverter
from meta_tl.linkage_result_revision import similarity_extension

data_sets = [
    # ('datasets/clean-A-1000.csv', 'datasets/clean-B-1000.csv', 'datasets/clean-true-matches-1000.csv'),
    # ('datasets/clean-A-10000.csv', 'datasets/clean-B-10000.csv', 'datasets/clean-true-matches-10000.csv'),
    # ('datasets/clean-A-100000.csv', 'datasets/clean-B-100000.csv', 'datasets/clean-true-matches-100000.csv'),
    # ('datasets/little-dirty-A-1000.csv', 'datasets/little-dirty-B-1000.csv',
    # 'datasets/little-dirty-true-matches-1000.csv'),
    # ('datasets/little-dirty-A-10000.csv', 'datasets/little-dirty-B-10000.csv',
    #  'datasets/little-dirty-true-matches-10000.csv'),
    # ('datasets/little-dirty-A-100000.csv', 'datasets/little-dirty-B-100000.csv', 'datasets/little-dirty-true-matches-100000.csv'),
    # ('datasets/very-dirty-A-1000.csv', 'datasets/very-dirty-B-1000.csv', 'datasets/very-dirty-true-matches-1000.csv'),
    ('datasets/very-dirty-A-10000.csv', 'datasets/very-dirty-B-10000.csv',
     'datasets/very-dirty-true-matches-10000.csv'),
    # ('datasets/very-dirty-A-100000.csv', 'datasets/very-dirty-B-100000.csv',
    #  'datasets/very-dirty-true-matches-100000.csv'),
]
is_efficient = True
for (ds_a, ds_b, gold) in data_sets:
    datasetA_name = ds_a
    datasetB_name = ds_b
    headerA_line = True  # Dataset A header line available - True or Flase
    headerB_line = True  # Dataset B header line available - True or Flase
    truthfile_name = gold
    # The two attribute numbers that contain the record identifiers
    #
    rec_idA_col = 0
    rec_idB_col = 0
    # The list of attributes to be used either for blocking or linking
    #
    #
    #  0: rec_id
    #  1: first_name
    #  2: middle_name
    #  3: last_name
    #  4: gender
    #  5: current_age
    #  6: birth_date
    #  7: street_address
    #  8: suburb
    #  9: postcode
    # 10: state
    # 11: phone
    # 12: email

    attrA_list = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11]
    attrB_list = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11]

    # The list of attributes to use for blocking (all must occur in the above
    # attribute lists)
    #
    blocking_funct_listA = [(blocking_functions.simple_blocking_key, 10), (blocking_functions.simple_blocking_key, 4)]
    blocking_funct_listB = [(blocking_functions.simple_blocking_key, 10), (blocking_functions.simple_blocking_key, 4)]

    # The list of tuples (comparison function, attribute number in record A,
    # attribute number in record B)
    #
    exact_comp_funct_list = [(string_functions.exact_comp, 1, 1),  # First name
                             (string_functions.exact_comp, 4, 4),  # Middle name
                             (string_functions.exact_comp, 3, 3),  # Last name
                             (string_functions.exact_comp, 8, 8),  # Suburb
                             (string_functions.exact_comp, 10, 10),  # State
                             ]
    comp_atts = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12]
    approx_comp_funct_list = [
        (string_functions.jaccard_comp, 3, 3),  # last name
        (string_functions.exact_comp, 4, 4),  # gender name
        (string_functions.jaccard_comp, 6, 6),  # birth date
        # (string_functions.jaccard_comp, 7, 7),  # street_address 0.869 F1-Score
        # (string_functions.jaccard_comp, 8, 8),  # First name, 0.94
        # (string_functions.edit_dist_sim_comp, 9, 9),  # postconde
        # (string_functions.edit_dist_sim_comp, 12, 12),  # email
    ]

    # =============================================================================
    #
    # Step 1: Load the two datasets from CSV files

    start_time = time.time()

    recA_dict, header_a_list = loadDataset.load_data_set(datasetA_name, rec_idA_col, \
                                                         attrA_list, headerA_line)
    recB_dict, header_b_list = loadDataset.load_data_set(datasetB_name, rec_idB_col, \
                                                         attrB_list, headerB_line)

    # Load data_io set of true matching pairs
    #
    true_match_set = loadDataset.load_truth_data(truthfile_name)
    weight_vector = threshold_classification.automatic_weight_computation(recA_dict, recB_dict, comp_atts)
    print(weight_vector)
    loading_time = time.time() - start_time

    # -----------------------------------------------------------------------------
    # Step 2: Block the datasets

    start_time = time.time()

    # Select one blocking technique

    # No blocking (all records in one block)
    #
    # blockA_dict = blocking.noBlocking(recA_dict)
    # blockB_dict = blocking.noBlocking(recB_dict)
    candidate_blocking_keys = blocking_funct_listA
    print("selected key")
    # for eps in [0.01, 0.02, 0.05]:
    #     candidate_blocking_keys = []
    #     for idx in attrA_list:
    #         candidate_blocking_keys.append((blocking_functions.phonetic_blocking_key, idx))
    #         candidate_blocking_keys.append((blocking_functions.simple_blocking_key, idx))
    #     # candidate_blocking_keys = blocking_key_selection.select_blocking_keys(recA_dict, recB_dict, candidate_blocking_keys,
    #     #                                                                       true_match_set, 200,
    #     #                                                                       0.02, 0.05)
    #
    #     candidate_blocking_keys = blocking_key_selection.select_blocking_keys(recA_dict, recB_dict,
    #                                                                           candidate_blocking_keys,
    #                                                                           true_match_set, 200,
    #                                                                           eps, 0.05)
    # print(eps)
    # for c in candidate_blocking_keys:
    #     print(c)
    # conjunctive blocking scheme
    attribute_qgram = QGramInstanceMatching(qgram=2)
    a_values = list(recA_dict.values())[0]
    b_values = list(recB_dict.values())[0]
    att_a_indices = [i for i in range(1, len(a_values))]
    att_b_indices = [i for i in range(1, len(b_values))]
    attribute_pairs = attribute_qgram.generate_attribute_pairs(recA_dict, recB_dict, att_a_indices, att_b_indices, 0.1,
                                                               False, is_top=True)
    blockA_dict = blocking.conjunctive_block(recA_dict, candidate_blocking_keys)
    blockB_dict = blocking.conjunctive_block(recB_dict, candidate_blocking_keys)
    blocking_time = time.time() - start_time

    # Print blocking statistics
    #
    blocking.print_block_statistics(blockA_dict, blockB_dict)

    # -----------------------------------------------------------------------------
    # Step 3: Compare the candidate pairs

    start_time = time.time()
    is_efficient = True
    if is_efficient:
        converter = QgramConverter()
        recA_dict = converter.convert_to_qgrams(recA_dict, att_a_indices, True, 2)
        recB_dict = converter.convert_to_qgrams(recB_dict, att_b_indices, True, 2)
        string_functions.is_efficient = is_efficient
    sim_vec_dict = comparison.compare_blocks(blockA_dict, blockB_dict, \
                                             recA_dict, recB_dict, \
                                             approx_comp_funct_list)

    sims = []
    for w in sim_vec_dict.values():
        sims.append(w)
    data_frame_2 = pd.DataFrame(data=np.asarray(sims),
                                columns=[str(i) for i in approx_comp_funct_list])
    sns.histplot(data=data_frame_2, bins=10, multiple='dodge')
    plt.show()
    comparison_time = time.time() - start_time
    sim_stats = []
    for v in sim_vec_dict.values():
        sim_stats.append(v)
    sim_stats = np.asarray(sim_stats)
    # -----------------------------------------------------------------------------
    # Step 4: Classify the candidate pairs

    start_time = time.time()
    atts = ["first_name", "middle_name", "last_name", "gender", "birth_date",
            "street_address", "suburb", "postcode", "state"]
    print(list(zip(atts, weight_vector)))
    sim_threshold = 0.5
    # active learning
    al_classification = ActiveLearningBootstrap(budget=500, iteration_budget=20, k=50)
    cal_model = al_classification.select_training_data(sim_vec_dict, true_match_set, constants.RF)
    class_match_set, class_nonmatch_set, pair_confidence = active_learning_solution.classify(sim_vec_dict, cal_model)
    classification_time = time.time() - start_time

    # -----------------------------------------------------------------------------
    # Step 5: Evaluate the classification

    # Get the number of record pairs compared
    #
    num_comparisons = len(sim_vec_dict)

    # Get the number of total record pairs to compared if no blocking used
    #
    all_comparisons = len(recA_dict) * len(recB_dict)

    # Get the list of identifiers of the compared record pairs
    #
    cand_rec_id_pair_list = sim_vec_dict.keys()

    # Blocking evaluation
    #
    rr = evaluation.reduction_ratio(num_comparisons, all_comparisons)
    pc = evaluation.pairs_completeness(cand_rec_id_pair_list, true_match_set)
    pq = evaluation.pairs_quality(cand_rec_id_pair_list, true_match_set)

    print('Blocking evaluation:')
    print('  Reduction ratio:    %.3f' % rr)
    print('  Pairs completeness: %.3f' % pc)
    print('  Pairs quality:      %.3f' % pq)
    print('')

    # Linkage evaluation
    #
    linkage_result = evaluation.confusion_matrix(class_match_set,
                                                 class_nonmatch_set,
                                                 true_match_set,
                                                 all_comparisons)
    accuracy = evaluation.accuracy(linkage_result)
    precision = evaluation.precision(linkage_result)
    recall = evaluation.recall(linkage_result)
    fmeasure = evaluation.fmeasure(linkage_result)
    print('Initial Linkage evaluation:')
    print('  Accuracy:    %.3f' % accuracy)
    print('  Precision:   %.3f' % precision)
    print('  Recall:      %.3f' % recall)
    print('  F-measure:   %.3f' % fmeasure)
    print('')
    average_sims = []
    sorted_scores, class_match_set = classification_result_analysis.determine_false_positives(class_match_set,
                                                                                              pair_confidence, False,
                                                                                              is_plot=True,
                                                                                              true_match_set=true_match_set)

    print("extend features")

    cand_comp_list = []
    base_atts = set([(c[0], c[1]) for c in approx_comp_funct_list])
    for a_p in attribute_pairs:
        if (a_p[0], a_p[1]) not in base_atts:
            cand_comp_list.append((string_functions.jaccard_comp, a_p[0], a_p[1]))
    reclassify_pairs, new_average_sims = similarity_extension.compute_additional_similarities(recA_dict, recB_dict,
                                                                                              sorted_scores,
                                                                                              cand_comp_list)
    print("{} pairs for reclassification".format(len(reclassify_pairs)))
    linkage_result = evaluation.confusion_matrix(class_match_set,
                                                 class_nonmatch_set,
                                                 true_match_set,
                                                 all_comparisons)
    accuracy = evaluation.accuracy(linkage_result)
    precision = evaluation.precision(linkage_result)
    recall = evaluation.recall(linkage_result)
    fmeasure = evaluation.fmeasure(linkage_result)
    print('Linkage evaluation without pairs for reclassification:')
    print('  Accuracy:    %.3f' % accuracy)
    print('  Precision:   %.3f' % precision)
    print('  Recall:      %.3f' % recall)
    print('  F-measure:   %.3f' % fmeasure)
    print('')
    selector = TopSelector(top_feature=2)
    indices = feature_distribution_analysis.get_additional_features(np.asarray(new_average_sims),
                                                                    feature_distribution_analysis.STD, selector)

    print(indices)
    reclassify_pairs_filtered = {}
    for p, vec in reclassify_pairs.items():
        unified_w = sim_vec_dict[p]
        filtered_w = [s for idx, s in enumerate(vec) if idx in indices]
        unified_w.extend(filtered_w)
        reclassify_pairs_filtered[p] = unified_w
    al_classification.budget = 100
    al_classification.iteration_budget = 10
    cal_model_extend = al_classification.select_training_data(reclassify_pairs_filtered, true_match_set, constants.RF)
    class_match_extend, class_nonmatch_set_extend, pair_confidence = active_learning_solution.classify(
        reclassify_pairs_filtered,
        cal_model_extend)
    data_frame_2 = pd.DataFrame(data=np.asarray(new_average_sims),
                                columns=[str(i) for i in cand_comp_list])
    sns.histplot(data=data_frame_2, bins=10, multiple='dodge')
    plt.show()
    class_match_set.update(class_match_extend)
    class_nonmatch_set.update(class_nonmatch_set_extend)
    linkage_result = evaluation.confusion_matrix(class_match_set,
                                                 class_nonmatch_set,
                                                 true_match_set,
                                                 all_comparisons)
    accuracy = evaluation.accuracy(linkage_result)
    precision = evaluation.precision(linkage_result)
    recall = evaluation.recall(linkage_result)
    fmeasure = evaluation.fmeasure(linkage_result)

    print('overall Linkage evaluation after extension:')
    print('  Accuracy:    %.3f' % accuracy)
    print('  Precision:   %.3f' % precision)
    print('  Recall:      %.3f' % recall)
    print('  F-measure:   %.3f' % fmeasure)
    print('')
    linkage_time = loading_time + blocking_time + comparison_time + \
                   classification_time
    print('Blocking runtime required for linkage: %.3f sec' % blocking_time)
    print('comparison runtime required for linkage: %.3f sec' % comparison_time)
    print('classification runtime required for linkage: %.3f sec' % classification_time)
    print('Total runtime required for linkage: %.3f sec' % linkage_time)

# -----------------------------------------------------------------------------

# End of program.
