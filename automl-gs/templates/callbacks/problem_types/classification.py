        logloss = log_loss(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred_label)
        precision = precision_score(y_true, y_pred_label, average='micro')
        recall = recall_score(y_true, y_pred_label, average='micro')
        f1 = f1_score(y_true, y_pred_label, average='micro')
        {% if problem_type == 'binary_classification' %}
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
        auc = metrics.auc(fpr, tpr)
        {% endif %}

        metrics = [logloss,
                   acc,
                   {% if problem_type == 'binary_classification' %}auc,{% endif %}
                   precision,
                   recall,
                   f1]