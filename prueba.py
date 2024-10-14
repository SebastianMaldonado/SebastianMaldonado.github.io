numeric_features_names = numeric_features
categorical_features_names = best_pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
all_feature_names = np.hstack([numeric_features_names, categorical_features_names])
feature_importances = best_model.feature_importances_

importance_df = pd.DataFrame({
    'Feature': all_feature_names,
    'Importance': feature_importances
})

importance_df = importance_df.sort_values(by='Importance', ascending=False)
country_importance_sum = importance_df[importance_df['Feature'].str.contains('Country')]['Importance'].sum()
status_importance_sum = importance_df[importance_df['Feature'].str.contains('Status_')]['Importance'].sum()
importance_df = importance_df[~importance_df['Feature'].str.contains('Country|Status_')]

sums_df = pd.DataFrame({
    'Feature': [
        'Country',
        'Status'
    ],
    'Importance': [
        country_importance_sum,
        status_importance_sum
    ]
})

importance_df = pd.concat([importance_df, sums_df], ignore_index=True)
importance_df = importance_df.sort_values(by='Importance', ascending=False)
importance_df