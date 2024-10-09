import argparse
import os
import time
import warnings
from statistics import mean

import numpy as np
from store_pipeline.incremental import incremental_boost
from store_pipeline.incremental.cluster_selection import determine_best_cluster
from store_pipeline.incremental.util import split_linkage_problem_tasks, \
    split_linkage_problem_tasks_on_training_data_pairs
from store_pipeline.statistic import statistical_tests
from numpy import std

from data_io import linkage_problem_io
from data_io.test_data import reader, wdc_reader, almser_linkage_reader
from record_linkage.classification.machine_learning import constants, active_learning_solution
from record_linkage.classification.machine_learning.active_learning_solution import ActiveLearningBootstrap
from record_linkage.evaluation import metrics
from store_pipeline.linkage_problem_analysis import util

# Ignore all warnings
warnings.filterwarnings("ignore")

MAIN_PATH = os.getcwd()

# Local application imports
from store_pipeline.utils import (
    add_task, add_singleton_task, create_graph, prepare_numpy_to_similarity_comparison_from_lp,
)
from store_pipeline.statistic.statistical_tests import compute_similarity_test_numpy
from graph_clustering import detect_communities
from model_selection import select_linkage_tasks_from_communities
from model_generation import generate_models
from evaluation import evaluate_predictions, evaluate_prediction_per_model

IS_BOOST = False

parser = argparse.ArgumentParser(description='rl generation')

parser.add_argument('--data_file', '-d', type=str, default='datasets/dexter/DS-C0/SW_0.3', help='data file')
# parser.add_argument('--train_pairs', '-tp', type=str, default='data/linkage_problems/wdc_almser/train_pairs_fv.csv',
#                     help='train pairs')
# parser.add_argument('--test_pairs', '-gs', type=str, default='data/linkage_problems/wdc_almser/test_pairs_fv.csv',
#                     help='test pairs')
parser.add_argument('--train_pairs', '-tp', type=str, default='data/linkage_problems/music_almser/train_pairs_fv.csv',
                    help='train pairs')
parser.add_argument('--test_pairs', '-gs', type=str, default='data/linkage_problems/music_almser/test_pairs_fv.csv',
                    help='test pairs')

# parser.add_argument('--linkage_tasks_dir', '-l', type=str, default='data/linkage_problems/wdc_almser',
#                     help='linkage problem directory')
# parser.add_argument('--linkage_tasks_dir', '-l', type=str, default='data/linkage_problems/dexter',
#                     help='linkage problem directory')
parser.add_argument('--linkage_tasks_dir', '-l', type=str, default='data/linkage_problems/music_almser',
                    help='linkage problem directory')
parser.add_argument('--statistical_test', '-s', type=str, default='ks_test',
                    choices=['ks_test', 'wasserstein_distance', 'calculate_psi'],
                    help='statistical test for comparing lps')
parser.add_argument('--ratio_sim_atomic_dis', '-rs', type=int, default=0,
                    help='amount of similar feature distributions so the lps are considered as similar')
parser.add_argument('--comm_detect', '-cd', type=str, default='leiden',
                    choices=['leiden', 'girvan_newman', 'label_propagation_clustering', 'louvain'],
                    help='communitiy detection algorithm')
parser.add_argument('--relevance_score', '-re', type=str, default='betweenness_centrality',
                    choices=['betweenness_centrality','largest', 'pageRank'],
                    help='relevance score for ordering the linkage problems in a cluster')
parser.add_argument('--active_learning', '-al', type=str, default='almser', choices=['almser', 'bootstrap'],
                    help='active learning algorithm')
parser.add_argument('--min_budget', '-mb', type=int, default=50,
                    help='minimum budget for each cluster')
parser.add_argument('--total_budget', '-tb', type=int, default=1000,
                    help='total budget')
parser.add_argument('--budget_unsolved', '-ub', type=int, default=200,
                    help='budget for unsolved linkage problems being not similar to any solved one')
parser.add_argument('--batch_size', '-b', type=int, default=5,
                    help='batch size')

args = parser.parse_args()
tps_list, fps_list, fn_list, p_list, r_list, f_list, run_t_list = [], [], [], [], [], [], []

# 'ks_test', 'wasserstein_distance', 'calculate_psi', 'ML_based', 'MMD'
STATISTICAL_TEST = args.statistical_test
FEATURE_CASE = 2  # 1 for all features to have the same distribution, 2 for majority of features to have the same distributions
RATIO_SIM_ATOMIC_DIS = args.ratio_sim_atomic_dis
COMMUNITY_DETECTION_ALGORITHM = args.comm_detect  # or leiden, girvan_newman, 'label_propagation_clustering', louvain
ACTIVE_LEARNING_ALGORITHM = args.active_learning  #bootstrap, almser, QHC
ACTIVE_LEARNING_MIN_BUDGET = args.min_budget
ACTIVE_LEARNING_ITERATION_BUDGET = args.batch_size
ACTIVE_LEARNING_TOTAL_BUDGET = args.total_budget
ACTIVE_LEARNING_MIN_UNSOLVED = args.budget_unsolved
ACTIVE_LEARNING_ITERATION_BUDGET_UNSOLVED = args.batch_size
SELECTION_STRATEGY = args.relevance_score  # betweenness_centrality, largest, pageRank
multivariate = STATISTICAL_TEST in ['ML_based', 'MMD']

for i in range(3):
    print(args.linkage_tasks_dir)
    unsupervised_gold_links = set()
    if 'dexter' in args.linkage_tasks_dir:
        file_name = os.path.join(MAIN_PATH, args.data_file)
        entities, _, _ = reader.read_data(file_name)
        gold_clusters = reader.generate_gold_clusters(entities)
        gold_links = metrics.generate_links(gold_clusters)
        data_sources_dict, data_sources_headers = reader.transform_to_data_sources(entities)
    elif 'wdc_computer' in args.linkage_tasks_dir:
        train_tp_links, train_tn_links, test_tp_links, test_tn_links = wdc_reader.read_wdc_links(args.train_pairs,
                                                                                                 args.test_pairs)
        gold_links = set()
        gold_links.update(train_tp_links)
        gold_links.update(test_tp_links)
    elif 'wdc_almser' in args.linkage_tasks_dir or 'music_almser' in args.linkage_tasks_dir:
        gold_links = set()
        train_tp_links, train_tn_links, test_tp_links, test_tn_links, unsup_train_tp_links, unsup_train_tn_links = (
            almser_linkage_reader.read_wdc_links(args.train_pairs, args.test_pairs))
        gold_links.update(train_tp_links)
        gold_links.update(test_tp_links)
        unsupervised_gold_links.update(unsup_train_tp_links)
        print("tps overall {}".format(len(gold_links)))
    unsupervised_gold_links = set()
    RECORD_LINKAGE_TASKS_PATH = os.path.join(MAIN_PATH, args.linkage_tasks_dir)
    ML_MODEL = constants.RF
    data_source_comp: dict[(str, str):[dict[(str, str):list]]] = linkage_problem_io.read_linkage_problems(
        RECORD_LINKAGE_TASKS_PATH, deduplication=False)
    all_pairs = set([p for lp in data_source_comp.values() for p in lp.keys()])
    # ===================================================
    # Step 1: Prepare Record Linkage Tasks
    # ===================================================

    reduced_comp = linkage_problem_io.remove_empty_problems(data_source_comp)

    # Count and print the total number of record pairs across all tasks
    total_record_pairs = util.count_total_number_of_links(reduced_comp)
    print(f"Total number of record pairs: {total_record_pairs}")

    # ===================================================
    # Step 2: Perform Linkage Tasks Distribution Test
    # ===================================================
    start_overall_time = time.time()

    linkage_problems = [(k[0], k[1], lp) for k, lp in data_source_comp.items()]
    relevant_columns = [col_index for col_index in range(len(list(linkage_problems[0][2].values())[0]))]
    print("relevant columns {}".format(relevant_columns))
    # split linkage problems to solved problems and to unsolved
    if 'dexter' in args.linkage_tasks_dir:
        solved_problems, integrated_sources, unsolved_problems = split_linkage_problem_tasks(linkage_problems,
                                                                                             split_ratio=0.5,
                                                                                             is_shuffle=True)
    elif 'wdc_computer' in args.linkage_tasks_dir:
        train_check = set(train_tp_links)
        train_check.update(train_tn_links)
        test_check = set(test_tp_links)
        test_check.update(test_tn_links)
        print("test links:{}".format(len(test_check)))
        solved_problems, integrated_sources, unsolved_problems, data_source_comp = (
            split_linkage_problem_tasks_on_training_data_pairs
            (data_source_comp, train_check, test_check))
        removed_pairs = set(data_source_comp.keys()).difference(
            set([(t[0], t[1]) for t in solved_problems]).union([(t[0], t[1]) for t in unsolved_problems]))
        print("removed pairs {}".format(len(removed_pairs)))
        for t in removed_pairs:
            del data_source_comp[t]
        assert len(solved_problems) + len(unsolved_problems) == len(data_source_comp), (str(len(data_source_comp)) +
                                                                                        "  " + str(
                    len(solved_problems) + len(unsolved_problems)))
    elif 'wdc_almser' in args.linkage_tasks_dir or 'music_almser' in args.linkage_tasks_dir:
        #data_source_comp = wdc_linkage_reader.split_linkage_problems(args.train_pairs, args.test_pairs, data_source_comp)
        solved_problems = []
        unsolved_problems = []
        integrated_sources = set()
        tps_check = 0
        for lp, sims in data_source_comp.items():
            if 'train' in lp[0]:
                solved_problems.append((lp[0], lp[1], sims))
                for p in sims.keys():
                    if p in gold_links:
                        tps_check += 1
                integrated_sources.add(lp[0].replace('_train', ''))
                integrated_sources.add(lp[1].replace('_train', ''))
            if 'test' in lp[0]:
                unsolved_problems.append((lp[0], lp[1], sims))
        print("number of tps in lps {}".format(tps_check))

    weights = []
    linkage_problems_numpy_arrays = [prepare_numpy_to_similarity_comparison_from_lp(task[2]) for task in
                                     solved_problems]
    all_sims = np.vstack(linkage_problems_numpy_arrays)
    print("all vectors {}".format(all_sims.shape))
    weights = np.std(all_sims, axis=0)
    weights[weights == 0] = 0.05
    print(weights)

    if not os.path.exists(
            os.path.join(RECORD_LINKAGE_TASKS_PATH, "incremental/statistical_tests_{}".format(STATISTICAL_TEST))):
        linkage_tasks_similarity_df, linkage_tasks_general_df = compute_similarity_test_numpy(
            STATISTICAL_TEST, solved_problems, relevant_columns, multivariate=multivariate, weights=weights,
            is_save=False,
            path=os.path.join(RECORD_LINKAGE_TASKS_PATH, 'incremental')
        )
    else:
        linkage_tasks_similarity_df, linkage_tasks_general_df = statistical_tests \
            .read_statistical_results(path=os.path.join(RECORD_LINKAGE_TASKS_PATH, 'incremental'),
                                      case=STATISTICAL_TEST)

    # ===================================================
    # Step 3: Perform Graph Clustering on Linkage Tasks
    # ===================================================

    # Create a graph of linkage tasks based on their similarity
    print(linkage_tasks_general_df)
    graph, task_mapping = create_graph(linkage_tasks_general_df, case=FEATURE_CASE,
                                       ratio_atomic_dis=RATIO_SIM_ATOMIC_DIS)

    # Record the start time for graph clustering
    node_labels = {v: v for k, v in task_mapping.items()}
    # Detect communities within the graph using the selected algorithm
    linkage_task_communities = detect_communities(
        COMMUNITY_DETECTION_ALGORITHM, graph
    )
    # ===================================================
    # Step 4: Select Linkage Tasks from Each Community
    # ===================================================

    # Select the largest task as representative as key and a order list of tasks based on a certain relevance criterion
    selected_tasks = select_linkage_tasks_from_communities(
        data_source_comp, linkage_task_communities, task_mapping,
        selection_strategy=SELECTION_STRATEGY, graph=graph
    )
    for tasks in selected_tasks.values():
        print("community: {}".format(tasks))

    # ===================================================
    # Step 5: Apply Active Learning to Label Selected Tasks
    # ===================================================

    # Record the start time for active learning
    start_time = time.time()

    # Apply active learning to label the selected tasks (uncomment when ready)
    cal_model_dict, selected_tasks, train_data_dict, used_lps_for_training = generate_models(
        selected_tasks, data_source_comp, linkage_tasks_general_df,
        min_budget=ACTIVE_LEARNING_MIN_BUDGET, iteration_budget=ACTIVE_LEARNING_ITERATION_BUDGET,
        total_budget=ACTIVE_LEARNING_TOTAL_BUDGET, gold_links=gold_links, unsup_gold_links=unsupervised_gold_links,
        model_name=ML_MODEL, active_learning_strategy=ACTIVE_LEARNING_ALGORITHM)

    print("Finished preparation for solved linkage problems")
    print("Start to solve new linkage problems")

    used_budgets_unsolved = 0
    result_dictionary = {}
    results_per_model = {}
    selected_numpy_dict = {}
    # ===================================================
    # Step 6.0: Prepare solved and unsolved tasks by initializing numpy arrays
    # Utilize selected training data feature vectors to determine similarity between cluster and unsolved linkage problem
    # ===================================================

    selected_numpy_dict = train_data_dict

    unsolved_numpy_dict = {}
    debug_labels = {}
    for unsolved_task in unsolved_problems:
        lp = unsolved_task[2]
        lp_numpy = statistical_tests.prepare_numpy_to_similarity_comparison_from_lp(lp)
        lp_labels = [tuple(sorted(k)) in gold_links for k in lp.keys()]
        unsolved_numpy_dict[(unsolved_task[0], unsolved_task[1])] = lp_numpy
        debug_labels[(unsolved_task[0], unsolved_task[1])] = np.asarray(lp_labels)

    # ===================================================
    # Step 6: Apply model search for unsolved problems
    # ===================================================
    while len(unsolved_numpy_dict) != 0:
        # select most similar task
        selected_task, unsolved_problem, linkage_task_sims = determine_best_cluster(selected_numpy_dict,
                                                                                    integrated_sources,
                                                                                    unsolved_numpy_dict,
                                                                                    relevant_columns,
                                                                                    multivariate, STATISTICAL_TEST,
                                                                                    ratio_sim_atomic_dis=RATIO_SIM_ATOMIC_DIS,
                                                                                    weights=weights)
        if unsolved_problem is not None:
            integrated_sources.add(unsolved_problem[0].replace('_test', ''))
            integrated_sources.add(unsolved_problem[1].replace('_test', ''))
        print(len(unsolved_numpy_dict))
        print("selection: {} for unsolved problem {}".format(selected_task, unsolved_problem))
        # apply model on unsolved model
        if selected_task is not None:
            unlabeled_class_match = set()
            unlabeled_non_class_match = set()
            model = cal_model_dict[selected_task]
            lp_problem = data_source_comp[unsolved_problem]
            graph = add_task(graph, linkage_task_sims, task_mapping, FEATURE_CASE, ratio_atomic_dis=0)
            if IS_BOOST:
                training_data, class_labels = train_data_dict[selected_task]
                new_training_data, new_training_labels = \
                    incremental_boost.boost_training_data(training_data, class_labels,
                                                          unsolved_numpy_dict[unsolved_problem], lp_problem,
                                                          debug_labels[unsolved_problem])
                if training_data.shape[0] != new_training_data.shape[0]:
                    print("pos/neg ratio: {} number of pairs {}"
                          .format(np.sum(class_labels) / class_labels.shape[0], class_labels.shape[0]))
                    print("boosted pos/neg ratio: {} number of pairs {}"
                          .format(np.sum(new_training_labels) / new_training_labels.shape[0],
                                  new_training_labels.shape[0]))
                    model = active_learning_solution.train_model(new_training_data, new_training_labels, ML_MODEL,
                                                                 False, True)
            class_match_set, class_non_match_set, pair_confidences = active_learning_solution \
                .classify(lp_problem, model)
            if selected_task not in results_per_model:
                results_per_model[selected_task] = set(), set(), {}
            model_class_matches, model_class_non_matches, pair_confidences_model = results_per_model[selected_task]
            model_class_matches.update(class_match_set)
            model_class_non_matches.update(class_non_match_set)
            pair_confidences_model.update(pair_confidences)

            class_match_set.union(unlabeled_class_match)
            class_non_match_set.union(unlabeled_non_class_match)
            del unsolved_numpy_dict[unsolved_problem]
        else:
            # if we found no similar cluster, we have to build a new one.
            activeLearning = ActiveLearningBootstrap(ACTIVE_LEARNING_MIN_UNSOLVED,
                                                     ACTIVE_LEARNING_ITERATION_BUDGET_UNSOLVED)
            train_vectors, train_class = activeLearning.select_training_data(data_source_comp[unsolved_problem],
                                                                             gold_links)
            used_budgets_unsolved += train_class.shape[0]
            model = active_learning_solution.train_model(train_vectors, train_class, ML_MODEL, False,
                                                         is_fine_tuned=True)
            print(unsolved_problem)
            cal_model_dict[unsolved_problem] = model
            class_match_set, class_non_match_set, pair_confidences = active_learning_solution \
                .classify(data_source_comp[unsolved_problem], model)
            graph = add_singleton_task(graph, unsolved_problem, task_mapping)
            selected_numpy_dict[unsolved_problem] = unsolved_numpy_dict[unsolved_problem]
            del unsolved_numpy_dict[unsolved_problem]
        result_dictionary[unsolved_problem] = (class_match_set, class_non_match_set, pair_confidences)
    overall_elapsed_time = time.time() - start_overall_time
    print(f"overall Elapsed time: {overall_elapsed_time:.2f} seconds")
    # ===================================================
    # Step 8: Evaluate the Results
    # ===================================================
    # Evaluate the performance of the labeled tasks

    model_results = evaluate_prediction_per_model(results_per_model, gold_links)
    for task, results in model_results.items():
        print(task)
        print("P={} R={} F1={}".format(results[3], results[4], results[5]))
    tps, fps, fns, p, r, f = evaluate_predictions(result_dictionary, gold_links)
    tps_list.append(tps)
    fps_list.append(fps)
    fn_list.append(fns)
    p_list.append(p)
    r_list.append(r)
    f_list.append(f)
    run_t_list.append(overall_elapsed_time)
data_set_name = ''
if 'dexter' in args.linkage_tasks_dir:
    data_set_name = 'Dexter'
elif 'wdc_almser' in args.linkage_tasks_dir:
    data_set_name = 'WDC-computer'
elif 'music_almser' in args.linkage_tasks_dir:
    data_set_name = 'Music'
with open('results/trans_er_incremental_results.csv', 'a') as result_file:
    result_file.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(STATISTICAL_TEST,
                                                                                    FEATURE_CASE,
                                                                                    COMMUNITY_DETECTION_ALGORITHM,
                                                                                    False,
                                                                                    ACTIVE_LEARNING_TOTAL_BUDGET,
                                                                                    sum(tps_list) + sum(fn_list),
                                                                                    sum(tps_list),
                                                                                    sum(fps_list),
                                                                                    sum(fn_list),
                                                                                    f"{mean(p_list):.3f}",
                                                                                    f"{mean(r_list):.3f}",
                                                                                    f"{mean(f_list):.3f}",
                                                                                    f"{std(f_list):.3f}",
                                                                                    f"{mean(run_t_list):.3f}",
                                                                                    SELECTION_STRATEGY,
                                                                                    data_set_name,
                                                                                    ACTIVE_LEARNING_ALGORITHM))
    result_file.close()
