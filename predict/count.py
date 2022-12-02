def count(df_pred):
    neg_count = 0
    neutral_count = 0
    pos_count = 0

    for i in df_pred:
        if i == 'D':
            pos_count += 1
        elif i == 'N':
            neutral_count += 1
        elif i == 'TD':
            neg_count += 1

    return [['Sentiment', 'Jumlah'], ['Mendukung', pos_count], ['Netral', neutral_count], ['Tidak Mendukung', neg_count]]
