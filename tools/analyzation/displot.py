import plotly.figure_factory as ff
import numpy as np
import pandas as pd
import os
import ast



def create_distplot(hist_data, group_labels, colors):
    assert len(hist_data) == len(group_labels) == len(colors)

    means = [np.mean(x) for x in hist_data]  # Calculate mean values
    #round the means to 2 decimal places
    means = [round(x, 2) for x in means]

    y_vals = [1.321, 1.595, 1.505, 2.08]

    fig = ff.create_distplot(hist_data, group_labels, show_hist=False, colors=colors, show_rug=False)

    # Add mean lines
    for i, mean in enumerate(means):
        fig.add_shape(type="line",
                    x0=mean,
                    y0=0,
                    x1=mean,
                    y1=y_vals[i],
                    line=dict(
                        color=colors[i],
                        width=2,
                        dash="dot",
                    ))

    # Update layout for the plot background and frame
    fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', legend=dict(
            x=0.2,  # x=0 and y=1 places the legend at the top left
            y=1,
            traceorder="normal",
            font=dict(
                family="sans-serif",
                size=16,
                color="black"
            ),
            bgcolor="rgba(255,255,255,0.5)",  # Slightly transparent white background
            bordercolor="Black",
            borderwidth=1
        )
    )
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror='allticks',
                    tickmode='array', tickvals=[0,*means,1], ticktext=[0, means[0], means[1], means[2], means[3], 1],
                    range=[min(0, min(means)-0.1), max(1, max(means)+0.1)],
                    tickfont=dict(size=16))
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror='allticks', showticklabels=False)
    # save with tight layout
    fig.write_image("seen_distribution.png")
    # also save the fig as html
    fig.write_html("seen_distribution.html")
    # also save as svg file
    fig.write_image("seen_distribution.svg")


if __name__ == "__main__":
    test = False
    analyzation_path = ['i3040_newdetector_50p', 'i3040_newdetector_100p', 'i3040_newdetector_150p' ,'i3040_newdetector_200p']
    if test:
        x1 = np.random.randn(200) - 1
        x2 = np.random.randn(200)
        x3 = np.random.randn(200) + 1

        hist_data = [x1, x2, x3]


        group_labels = ['Group 1', 'Group 2', 'Group 3']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

        create_distplot(hist_data, group_labels, colors)
    
    else:
        hist_data = []
        group_labels = ['p=5%', 'p=10%', 'p=15%', 'p=20%']
        colors = colors = ['#D3E5FF', '#A1C9FD', '#5A8FDB', '#2E5A88']
        for path in analyzation_path:
            # load the dataset
            dataset = pd.read_csv(os.path.join('data', path, 'dataset.csv'))
            dataset['complete_vehicle_infos'] = dataset['complete_vehicle_infos'].apply(ast.literal_eval)
            dataset['detected_vehicle_infos'] = dataset['detected_vehicle_infos'].apply(ast.literal_eval)
            dataset['fco_vehicle_infos'] = dataset['fco_vehicle_infos'].apply(ast.literal_eval)

            dataset['len_fco'] = dataset['fco_vehicle_infos'].apply(lambda x: len(x))
            dataset['len_detected'] = dataset['detected_vehicle_infos'].apply(lambda x: len(x))

            dataset['len_complete'] = dataset['complete_vehicle_infos'].apply(lambda x: len(x))

            # create new column with all vehicles that are in the fco or detected without duplicates
            dataset['fco_or_detected'] = dataset.apply(lambda x: list(set(list(x['fco_vehicle_infos'].keys()) + list(x['detected_vehicle_infos'].keys()))), axis=1)
            dataset['fco_or_detected_len'] = dataset['fco_or_detected'].apply(lambda x: len(x))

            dataset['seen_percentage'] = dataset['fco_or_detected_len'] / dataset['len_complete']

            hist_data.append(dataset['seen_percentage'].values)

        create_distplot(hist_data, group_labels, colors)



