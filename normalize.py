

for i in range(self._sinkorn_layers):
    groups_1 = tf.unsorted_segment_sum(edges_ex, group_indices_1, num_groups_1)
    normalize_1 = tf.gather(groups_1, group_indices_1)
    edges_ex = edges_ex / (normalize_1 + 1e-8)

    groups_2 = tf.unsorted_segment_sum(edges_ex, group_indices_2, num_groups_2)
    normalize_2 = tf.gather(groups_2, group_indices_2)
    edges_ex = edges_ex / (normalize_2 + 1e-8)