class meta_callback(Callback):
    """Keras Callback used during model training to save current weights
    and logs after each training epoch.
    """

    def on_train_begin(self, logs={}):
        self.f = open('metadata/results.csv', 'w')
        self.w= csv.writer(self.f)
        self.w.writerow(['epoch'])

    def on_train_end(self, logs={}):
        self.f.close()

    def on_epoch_end(self, epoch, logs={}):
        y_true = self.model.validation_data[1]
        y_pred = self.model.predict(self.model.validation_data[0])

        {% include 'callbacks/problem_types/' ~ problem_type ~ '.py' %}

        self.w.writerow([epoch+1] + metrics)

        # Only run while using automl-gs, which tells it an epoch is finished
        # and data is recorded.
        if args['context'] == 'automl-gs':
            sys.stdout.flush()
            print("EPOCH_END")