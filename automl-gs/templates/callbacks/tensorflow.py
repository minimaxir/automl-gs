class meta_callback(Callback):
    """Keras Callback used during model training to save current weights
    and logs after each training epoch.
    """

    def __init__(self, args):
        self.f = open(os.path.join('metadata', 'results.csv'), 'w')
        self.w= csv.writer(self.f)
        self.w.writerow(['epoch', 'time_completed'] + {{ metrics }})
        self.in_automl = args.context == 'automl-gs'

    def on_train_end(self, logs={}):
        self.f.close()

    def on_epoch_end(self, epoch, logs={}):
        print(dir(self.model))
        y_true = self.validation_data[1]
        y_pred = self.model.predict(self.validation_data[0])

        {% include 'callbacks/problem_types/' ~ problem_type ~ '.py' %}

        time_completed = "{:%Y-%m-%d %H:%M:%S}".format(datetime.utcnow())
        self.w.writerow([epoch+1, time_completed] + metrics)

        # Only run while using automl-gs, which tells it an epoch is finished
        # and data is recorded.
        if self.in_automl:
            sys.stdout.flush()
            print("EPOCH_END")