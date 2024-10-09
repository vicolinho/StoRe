from baseline.almser.ALMSER import *


class ALMSER_EXP(ALMSER):
    pass

    def run_AL(self, is_batch=False):
        print("Start ALMSER")
        start_time = time.time()
        print_progress(1, self.quota, prefix="ALMSER Mode: Active Learning")
        initial_qs = self.query_strategy
        initial_criteria = self.criteria
        result_row = 0
        for i in range(0, self.quota, self.batch_size):
            self.iteration = i
            self.simplify_qs(i, initial_qs, initial_criteria)
            print(self.query_strategy)
            self.criteria = get_criteria_list(self.query_strategy)
            self.update_criteria_scores(i)
            self.get_informativeness_score(i)
            if is_batch:
                print(i)
                batch_size = min(self.quota-i, self.batch_size)
                candidate, strategy = self.select_pair_batch(batch_size)
            else:
                candidate, strategy = self.select_pair()
            if is_batch:
                s_id, t_id, true_label = self.update_after_answer_batch(candidate)
            else:
                s_id, t_id, true_label = self.update_after_answer(candidate)
                self.results.loc[result_row].Query = s_id + "-" + t_id
                self.results.loc[result_row].Query_Label = true_label
                self.results.loc[result_row].Strategy = strategy
                self.results.loc[result_row].Labeled_Set_Size = self.labeled_set.shape[0]
            print_progress(i + 1, self.quota, prefix="ALMSER Mode: Active Learning")
            # if self.quota - len(self.labeled_set.index) == 0:
            #     break
        elapsed_time = time.time() - start_time
        self.results.loc[result_row].run_time = elapsed_time
        if 'all' in self.learning_models and self.gs is not None:
            self.evaluateCurrentModel(result_row)
        result_row += 1

    def simplify_qs(self, i, initial_qs, initial_criteria):
        # do the basic for first iterations
        if ('graph_signal' in initial_criteria) or ('ensemble_graph' in initial_criteria):
            if i >= 20:
                self.query_strategy = initial_qs
            else:
                self.query_strategy = 'disagreement'

    def update_criteria_scores(self, iteration):
        # try:

        if 'uncertainty' in self.criteria:
            unlabeled_data = self.get_feature_vector_subset(self.unlabeled_set, getLabels=False)
            self.unlabeled_set['uncertainty'] = self.calculate_uncertainty(unlabeled_data)

        if 'predicted_label' in self.criteria:
            unlabeled_data = self.get_feature_vector_subset(self.unlabeled_set, getLabels=False)
            self.unlabeled_set['predicted_label'] = self.learning_models['all'].predict(unlabeled_data)
            pre_proba_both_classes = self.learning_models['all'].predict_proba(unlabeled_data)
            max_proba = map(lambda x: max(x), pre_proba_both_classes)
            self.unlabeled_set['pre_proba'] = list(max_proba)

        if 'disagreement' in self.criteria:
            if self.labeled_set.label.nunique() == 2:
                unlabeled_data = self.get_feature_vector_subset(self.unlabeled_set, getLabels=False)
                dis, votes = self.calculate_disagreement(unlabeled_data, ('graph_signal' in self.criteria),
                                                         ('correct_prediction' in self.criteria))
                self.unlabeled_set['disagreement'] = dis
                self.unlabeled_set['votes'] = votes
        if 'ensemble_graph' in self.criteria:
            predictions = self.unlabeled_set['predicted_label']
            self.unlabeled_set['majority_vote'] = predictions
            self.G = constructGraphFromWeightedPredictions(predictions, self.labeled_set, self.unlabeled_set, True,
                                                           self.unlabeled_set_metadata, self.count_sources)
            self.calculate_graph_info()

            self.unlabeled_set['graph_cc_size'] = self.unlabeled_set.apply(
                lambda row, G=self.G: len(nx.node_connected_component(G, row.source)) if has_path_(G, row.source,
                                                                                                   row.target) else 0,
                axis=1)

            self.unlabeled_set['sel_proba'] = self.unlabeled_set.apply(
                lambda row, G=self.G: 1 / row.graph_cc_size if row.graph_inferred_label else 1.0, axis=1)

            # log accuracy of all graph labels
            self.log.graph_log(self)
            # log accuracy of clean components (additional training data) labels
            self.log.boost_data_log(self)
            self.log.cc_distribution_log(self)
            self.log.pred_graph_diff_log(self)

        if 'disagreement_graph_pred' in self.criteria:

            self.unlabeled_set['disagreement_graph_pred'] = self.unlabeled_set.apply(
                lambda x, tasks_count=self.count_sources: 1 if (x['majority_vote'] != x[
                    'graph_inferred_label']) and (x.graph_cc_size <= tasks_count) else 0, axis=1)

            # if there is no disagreement between predictions and graph calculate committee disagreement
            if (self.unlabeled_set['disagreement_graph_pred'] == 0).all() and self.query_strategy == 'almser_gb':
                unlabeled_data = self.get_feature_vector_subset(self.unlabeled_set, getLabels=False)
                dis, votes = self.calculate_disagreement(unlabeled_data, ('graph_signal' in self.criteria),
                                                         ('correct_prediction' in self.criteria))
                self.unlabeled_set['disagreement'] = dis

        if 'datasource_pair_frequency' in self.criteria:
            if (self.labeled_set.shape[0] > 0):
                history_status = self.get_labeledset_current_status()

                for ds_p in history_status.keys():
                    ds_p_inx = self.unlabeled_set[self.unlabeled_set.datasource_pair == ds_p].index
                    self.unlabeled_set.loc[ds_p_inx, 'datasource_pair_frequency'] = history_status.get(ds_p)

        if 'task_rltd' in self.criteria:
            self.unlabeled_set['task_rltd'] = self.unlabeled_set_metadata.total_weight.values + 0.001


        # except Exception as e:
        #     print(str(e))
        #     import pdb
        #     pdb.set_trace()

    def get_informativeness_score(self, it_):

        self.unlabeled_set['inf_score'] = 0

        if self.query_strategy == "uncertainty" or self.query_strategy == "margin_boost_learner":
            self.unlabeled_set['inf_score'] = self.unlabeled_set['uncertainty'] * (-1)

        elif self.query_strategy == "disagreement_stratified" or self.query_strategy == "disagreement_graph_stratified":

            self.unlabeled_set['inf_score'] = (
                    self.unlabeled_set['disagreement'] * (1 - self.unlabeled_set['datasource_pair_frequency']))


        elif self.query_strategy == 'disagreement' or self.query_strategy == 'disagreement_post_graph' or self.query_strategy == 'rltd_disagreement':
            self.unlabeled_set['inf_score'] = self.unlabeled_set['disagreement']

        elif self.query_strategy == 'random_stratified':
            self.unlabeled_set['inf_score'] = 1 - self.unlabeled_set['datasource_pair_frequency']

        elif self.query_strategy == 'graph_based':
            self.unlabeled_set['inf_score'] = self.unlabeled_set.apply(
                lambda x: 1 if (x['predicted_label'] != x['graph_inferred_label']) else 0, axis=1)

        elif self.query_strategy == 'almser_gb' or self.query_strategy == 'almser_gb_transfer' or self.query_strategy == 'almser_gb_explore_exploit' or self.query_strategy == 'almser_group':

            if (self.unlabeled_set['disagreement_graph_pred'] == 0).all():
                print("No disagreement between graph and predictions. Will consider QHC.")
                self.unlabeled_set['inf_score'] = self.unlabeled_set['disagreement']
                self.unlabeled_set['sel_proba'] = 1
            else:
                self.unlabeled_set['inf_score'] = self.unlabeled_set['disagreement_graph_pred']

                dis_match = self.unlabeled_set[
                    (self.unlabeled_set['inf_score'] == 1) & (self.unlabeled_set['graph_inferred_label'] == True)].index
                dis_non_match = self.unlabeled_set[(self.unlabeled_set['inf_score'] == 1) & (
                        self.unlabeled_set['graph_inferred_label'] == False)].index

                self.unlabeled_set['sel_proba'] = 0
                self.unlabeled_set.loc[dis_match, 'sel_proba'] = 1.0 / float(len(dis_match)) if len(
                    dis_match) > 0 else 0
                self.unlabeled_set.loc[dis_non_match, 'sel_proba'] = 1.0 / float(len(dis_non_match)) if len(
                    dis_non_match) > 0 else 0

            if self.query_strategy == 'almser_gb_transfer' or self.query_strategy == 'almser_group':

                gb_scores = copy.copy(self.unlabeled_set['inf_score'])

                print("Iteration: %i" % it_)
                if it_ >= 20 and it_ % 20 == 0:
                    print("--Recalculate best store_pipeline setting--")
                    self.tasks_to_exploit = self.get_best_transf_setting(showHeatmap=True)

                    print("Current Task distribution:")
                    check = self.labeled_set.datasource_pair.hist()
                    plt.xticks(rotation='vertical')
                    plt.show()

                print("Tasks to exploit: ", self.tasks_to_exploit)

                ds_p_inx = self.unlabeled_set[~self.unlabeled_set.datasource_pair.isin(self.tasks_to_exploit)].index
                self.unlabeled_set.loc[ds_p_inx, 'inf_score'] = 0

                for task_exploit in self.tasks_to_exploit:
                    task_idx = self.unlabeled_set[(self.unlabeled_set['datasource_pair'] == task_exploit)].index

                    task_idx_lfp = self.unlabeled_set[(self.unlabeled_set['datasource_pair'] == task_exploit) & (
                            self.unlabeled_set['graph_inferred_label'] == False)].index
                    task_idx_lfn = self.unlabeled_set[(self.unlabeled_set['datasource_pair'] == task_exploit) & (
                            self.unlabeled_set['graph_inferred_label'] == True)].index
                    # if a task to exploit has no pred-graph disagreement for both lfp and lfn, take the committee disagreement rescaled (1-0)
                    # if (self.unlabeled_set.loc[task_idx,'inf_score'] == 0).all():
                    if ((self.unlabeled_set.loc[task_idx_lfp, 'inf_score'] == 0).all()) or (
                            (self.unlabeled_set.loc[task_idx_lfn, 'inf_score'] == 0)).all():
                        print("No dis for task %s. Will consider QHC." % task_exploit)
                        self.unlabeled_set.loc[task_idx, 'inf_score'] = self.unlabeled_set.loc[
                                                                            task_idx, 'disagreement'] / \
                                                                        self.unlabeled_set['disagreement'].max()
                        self.unlabeled_set.loc[task_idx, 'sel_proba'] = 1

                    else:
                        # give selection probability scores based on the count of lfp and lfn per task
                        lfp_of_task = self.unlabeled_set[(self.unlabeled_set['inf_score'] == 1) & (
                                self.unlabeled_set['graph_inferred_label'] == False) & (self.unlabeled_set[
                                                                                            'datasource_pair'] == task_exploit)].index
                        lfn_of_task = self.unlabeled_set[(self.unlabeled_set['inf_score'] == 1) & (
                                self.unlabeled_set['graph_inferred_label'] == True) & (self.unlabeled_set[
                                                                                           'datasource_pair'] == task_exploit)].index

                        self.unlabeled_set.loc[lfp_of_task, 'sel_proba'] = 1 / lfp_of_task.shape[0] if \
                            lfp_of_task.shape[0] > 0 else 0
                        self.unlabeled_set.loc[lfn_of_task, 'sel_proba'] = 1 / lfn_of_task.shape[0] if \
                            lfn_of_task.shape[0] > 0 else 0

            if self.query_strategy == 'almser_gb_explore_exploit':
                if it_ in range(20) or it_ in range(40, 50) or it_ in range(70, 80) or it_ in range(100,
                                                                                                    110) or it_ in range(
                    130, 140) or it_ in range(160, 170) or it_ in range(190, 200):
                    self.phase = 'explore'
                else:
                    self.phase = 'exploit'

                print("Iteration: %i %s" % (it_, self.phase))

                if self.phase == 'exploit':
                    if it_ == 20 or it_ == 50 or it_ == 80 or it_ == 110 or it_ == 140 or it_ == 170:
                        print("Change to exploit phase on iteration: %i" % it_)
                        print("--Recalculate tasks to exploit--")
                        self.tasks_to_exploit = self.get_best_transf_setting(showHeatmap=True)

                        print("Current Task distribution:")
                        check = self.labeled_set.datasource_pair.hist()
                        plt.xticks(rotation='vertical')
                        plt.show()

                    print("Tasks to exploit: ", self.tasks_to_exploit)

                    # if a task to exploit has no pred-graph disagreement, take the committee disagreement rescaled (1-0)
                    for task_exploit in self.tasks_to_exploit:
                        task_idx = self.unlabeled_set[(self.unlabeled_set['datasource_pair'] == task_exploit)].index
                        if (self.unlabeled_set.loc[task_idx, 'inf_score'] == 0).all():
                            print("No dis for task %s. Will consider QHC." % task_exploit)
                            max_dis = self.unlabeled_set['disagreement'].max()
                            self.unlabeled_set.loc[task_idx, 'inf_score'] = self.unlabeled_set.loc[
                                                                                task_idx, 'disagreement'] / max_dis
                            self.unlabeled_set.loc[task_idx, 'sel_proba'] = 1
                        else:
                            self.unlabeled_set.loc[task_idx, 'sel_proba'] = 1 + self.unlabeled_set.loc[
                                task_idx, 'sel_proba']

                    ds_p_inx = self.unlabeled_set[~self.unlabeled_set.datasource_pair.isin(self.tasks_to_exploit)].index
                    # with stratification on exploit tasks
                    # self.unlabeled_set['inf_score'] = (self.unlabeled_set['inf_score']*(1-self.unlabeled_set['datasource_pair_frequency']))
                    self.unlabeled_set.loc[ds_p_inx, 'inf_score'] = 0

                    # adjust weights so that all exploit tasks are equally selected
                    task_freq_scores = dict()
                    for task_exploit in self.tasks_to_exploit:
                        task_freq_scores[task_exploit] = self.unlabeled_set[
                                                             (self.unlabeled_set.datasource_pair == task_exploit) & (
                                                                     self.unlabeled_set.inf_score == 1)].inf_score.sum() / \
                                                         self.unlabeled_set[
                                                             self.unlabeled_set.inf_score == 1].inf_score.sum()

                    self.unlabeled_set['sel_proba'] = self.unlabeled_set.apply(
                        lambda x, scs=task_freq_scores: x['sel_proba'] * (
                            1 - scs.get(x['datasource_pair']) if x['inf_score'] == 1 else 0), axis=1)

                else:
                    self.unlabeled_set['inf_score'] = (
                            self.unlabeled_set['inf_score'] * (1 - self.unlabeled_set['datasource_pair_frequency']))


        elif self.query_strategy == 'exploit_explore':
            if it_ in range(20) or it_ in range(40, 50) or it_ in range(70, 80) or it_ in range(100,
                                                                                                110) or it_ in range(
                130, 140) or it_ in range(160, 170) or it_ in range(190, 200):
                self.phase = 'explore'
            else:
                self.phase = 'exploit'

            if self.phase == 'exploit':
                print("Iteration: %i" % it_)
                if it_ == 20 or it_ == 50 or it_ == 80 or it_ == 110 or it_ == 140 or it_ == 170:
                    print("Change to exploit phase on iteration: %i" % it_)
                    print("--Recalculate tasks to exploit--")
                    self.tasks_to_exploit = self.get_best_transf_setting()

                print("Tasks to exploit: ", self.tasks_to_exploit)

                self.unlabeled_set['inf_score'] = self.unlabeled_set['disagreement']
                ds_p_inx = self.unlabeled_set[~self.unlabeled_set.datasource_pair.isin(self.tasks_to_exploit)].index
                self.unlabeled_set.loc[ds_p_inx, 'inf_score'] = 0

            else:
                self.unlabeled_set['inf_score'] = (
                        self.unlabeled_set['disagreement'] * (1 - self.unlabeled_set['datasource_pair_frequency']))


        elif self.query_strategy == 'random':
            self.unlabeled_set['inf_score'] = 0

        else:
            print("Unknown query strategy. Cannot calculate informativeness score.")
            import pdb;
            pdb.set_trace();

    def select_pair(self):
        if self.query_strategy != 'random' and not self.unlabeled_set['inf_score'].isnull().all():
            max_qs_score = self.unlabeled_set['inf_score'].max()

            all_max_candidates = copy.copy(self.unlabeled_set[(self.unlabeled_set['inf_score'] == max_qs_score)])
            candidate = random.choice(all_max_candidates.index)

            if self.query_strategy == 'almser_gb' or self.query_strategy == 'almser_gb_transfer' or self.query_strategy == 'almser_gb_explore_exploit':
                candidate = random.choices(all_max_candidates.index, weights=all_max_candidates.sel_proba, k=1)[0]

            if self.query_strategy == 'almser_group':

                intersection_exploit_tasks_max_cand_tasks = set(self.tasks_to_exploit).intersection(
                    set(self.unlabeled_set.loc[all_max_candidates.index, 'datasource_pair']))
                if (len(intersection_exploit_tasks_max_cand_tasks) > 0):
                    task = random.choices(list(intersection_exploit_tasks_max_cand_tasks), k=1)[0]

                    candidate = random.choices(all_max_candidates[all_max_candidates.datasource_pair == task].index,
                                               weights=all_max_candidates[
                                                   all_max_candidates.datasource_pair == task].sel_proba, k=1)[0]
                else:
                    # if there is no good candidate from the task to exploit fllback to pure disagreement
                    print("Pure disagreement fallback")
                    candidate = random.choice(all_max_candidates.index)

            if self.query_strategy == 'disagreement_post_graph' or self.query_strategy == 'exploit_explore':
                # do some post processing. pick with probability 50% what is corrected by the graph.
                pick_conflict = random.uniform(0, 1) > 0.5

                if pick_conflict:
                    sel_proba = all_max_candidates.apply(
                        lambda x: x['sel_proba'] if x['majority_vote'] != x['graph_inferred_label'] else 0, axis=1)
                    candidate = random.choices(all_max_candidates.index, weights=sel_proba, k=1)[0]


                else:
                    candidate = random.choices(all_max_candidates.index, k=1)[0]

            if self.query_strategy == 'rltd_disagreement':
                candidate = \
                    random.choices(all_max_candidates.index, weights=all_max_candidates['task_rltd'].values, k=1)[0]

            strategy = self.query_strategy
            if (strategy == 'exploit_explore'): strategy = strategy + "_" + self.phase

        else:
            # random candidate if selection strategy is null or if it cannot be calculated otherwise
            candidate = random.choice(self.unlabeled_set.index)
            strategy = 'random'

        return candidate, strategy

    def select_pair_batch(self, batch_size):
        if self.query_strategy != 'random' and not self.unlabeled_set['inf_score'].isnull().all():
            top_k = self.unlabeled_set.nlargest(batch_size, 'inf_score', keep='all')
            min_qs_score = top_k['inf_score'].min()
            all_max_candidates = copy.copy(self.unlabeled_set[(self.unlabeled_set['inf_score'] > min_qs_score)])
            rest = batch_size - len(all_max_candidates.index)
            min_k_candidates = copy.copy(self.unlabeled_set[(self.unlabeled_set['inf_score'] == min_qs_score)])
            candidate = list(all_max_candidates.index.tolist())
            if rest > 0:
                min_k_candidate = random.choices(min_k_candidates.index, k=rest)
                candidate.extend(min_k_candidate)
            else:
                candidate = random.choices(all_max_candidates.index, k=batch_size)
            if self.query_strategy == 'almser_gb' or self.query_strategy == 'almser_gb_transfer' or self.query_strategy == 'almser_gb_explore_exploit':
                candidate = random.choices(top_k.index, weights=top_k.sel_proba, k=batch_size)
            if self.query_strategy == 'almser_group':
                intersection_exploit_tasks_max_cand_tasks = set(self.tasks_to_exploit).intersection(
                    set(self.unlabeled_set.loc[top_k.index, 'datasource_pair']))
                if len(intersection_exploit_tasks_max_cand_tasks) > 0:
                    task_batch = min(len(intersection_exploit_tasks_max_cand_tasks), batch_size)
                    tasks = random.choices(list(intersection_exploit_tasks_max_cand_tasks), k=task_batch)

                    candidate = random.choices(top_k[top_k['datasource_pair'].isin(tasks)].index,
                                               weights=top_k[top_k['datasource_pair'].isin(tasks)].sel_proba, k=batch_size)
                else:
                    # if there is no good candidate from the task to exploit fllback to pure disagreement
                    print("Pure disagreement fallback")
                    candidate = random.choices(top_k.index, k=batch_size)

            if self.query_strategy == 'disagreement_post_graph' or self.query_strategy == 'exploit_explore':
                # do some post processing. pick with probability 50% what is corrected by the graph.
                pick_conflict = random.uniform(0, 1) > 0.5

                if pick_conflict:
                    sel_proba = all_max_candidates.apply(
                        lambda x: x['sel_proba'] if x['majority_vote'] != x['graph_inferred_label'] else 0, axis=1)
                    candidate = random.choices(top_k.index, weights=sel_proba, k=batch_size)
                else:
                    candidate = random.choices(top_k.index, k=batch_size)

            if self.query_strategy == 'rltd_disagreement':
                candidate = \
                    random.choices(top_k.index, weights=top_k['task_rltd'].values, k=1)[0]

            strategy = self.query_strategy
            if strategy == 'exploit_explore': strategy = strategy + "_" + self.phase

        else:
            # random candidate if selection strategy is null or if it cannot be calculated otherwise
            candidate = random.choices(self.unlabeled_set.index, k= batch_size)
            strategy = 'random'

        return candidate, strategy
