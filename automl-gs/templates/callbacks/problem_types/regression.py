        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r_2 = r2_score(y_true, y_pred)

        metrics = [mse, mae, r_2]