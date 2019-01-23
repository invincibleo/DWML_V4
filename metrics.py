import tensorflow as tf

# def covariance(vec1, vec2):
#     mean_vec1 = tf.metrics.mean(vec1)
#     mean_vec2 = tf.metrics.mean(vec2)
#     cov_vec1_vec2 = tf.metrics.mean((vec1 - mean_vec1)(vec2 - mean_vec2))
#     return cov_vec1_vec2
#
def concordance_cc2(prediction, ground_truth):
    """Defines concordance metric for model evaluation. 

    Args:
       prediction: prediction of the model.
       ground_truth: ground truth values.
    Returns:
       The concordance value.
    """
    with tf.name_scope('my_metrics'):
        names_to_values, names_to_updates = tf.contrib.metrics.aggregate_metric_map({
            'eval/mean_pred': tf.metrics.mean(prediction),
            'eval/mean_lab': tf.metrics.mean(ground_truth),
            'eval/cov_pred': tf.contrib.metrics.streaming_covariance(prediction, prediction),
            'eval/cov_lab': tf.contrib.metrics.streaming_covariance(ground_truth, ground_truth),
            'eval/cov_lab_pred': tf.contrib.metrics.streaming_covariance(prediction, ground_truth)
        })

        metrics = dict()
        for name, value in names_to_values.items():
            metrics[name] = value

        mean_pred = metrics['eval/mean_pred']
        var_pred = metrics['eval/cov_pred']
        mean_lab = metrics['eval/mean_lab']
        var_lab = metrics['eval/cov_lab']
        var_lab_pred = metrics['eval/cov_lab_pred']

        denominator = (var_pred + var_lab + tf.square(mean_pred - mean_lab))

        concordance_cc2 = (2 * var_lab_pred) / denominator

        # concordance_cc2 = tf.Print(concordance_cc2, [tf.shape(concordance_cc2), tf.shape(prediction), tf.shape(ground_truth)], 'Debug_: mean_pred_value:')
        # tf.print("Debug:", tf.shape(concordance_cc2), output_stream=sys.stdout)

    return concordance_cc2, names_to_values, [x for x in names_to_updates.values()]
