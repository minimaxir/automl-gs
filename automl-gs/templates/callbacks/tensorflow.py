class meta_callback(Callback):
    """Keras Callback used during model training to save current weights
    and logs after each training epoch.
    """

    def __init__(self, args, X_val, y_val):
        self.f = open(os.path.join('metadata', 'results.csv'), 'w')
        self.w= csv.writer(self.f)
        self.w.writerow(['epoch', 'time_completed'] + {{ metrics }})
        self.in_automl = args.context == 'automl-gs'
        self.X_val = X_val
        self.y_val = y_val

    def on_train_end(self, logs={}):
        self.f.close()
        self.model.save_weights('model_weights.hdf5')

    def on_epoch_end(self, epoch, logs={}):
        y_true = self.y_val
        y_pred = self.model.predict(self.X_val)

        {% if problem_type != 'regression' %}
        y_pred_label = np.zeros(y_pred.shape)
        y_pred_label[np.arange(y_pred.shape[0]), y_pred.argmax(axis=1)] = 1
        {% endif %}

        {% include 'callbacks/problem_types/' ~ problem_type ~ '.py' %}

        time_completed = "{:%Y-%m-%d %H:%M:%S}".format(datetime.utcnow())
        self.w.writerow([epoch+1, time_completed] + metrics)

        # Only run while using automl-gs, which tells it an epoch is finished
        # and data is recorded.
        if self.in_automl:
            sys.stdout.flush()
            print("EPOCH_END")