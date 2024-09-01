import matplotlib.pyplot as plt
import seaborn as sns


def univariate_analysis(data, instructions):
    columns = instructions.get("columns")
    plot_type = instructions.get("plot_type")
    plot_width = instructions.get("plot_size").get("width")
    plot_height = instructions.get("plot_size").get("height")

    for column in columns:
        plt.figure(figsize = (plot_width, plot_height))
        # By default we will build cat plots for all available fields        
        if plot_type == "pie":
            if data[column].dtype in (int, float):
                continue
            # Default
            plt.pie(
                x = data[column].value_counts(), labels = data[column].value_counts().keys(),
                autopct="%.2f"
            )
        elif plot_type == "hist":
            if data[column].dtype == "object":
                continue
            else:
                sns.histplot(data = data, x = column, kde = True)
        elif plot_type == "box":
            if data[column].dtype == "object":
                continue
            else:
                sns.boxplot(data = data, x = column)
        else:
            if data[column].dtype in (int, float):
                continue
            sns.countplot(data = data, x = column)
        
        plt.title(f"Univariate {plot_type} plot for {column}")
        plt.savefig(f"./plots/univariate_plots/{plot_type}_{column}.png")
    

def bi_variate_analysis(data, instructions):
    # We are always expecting two columns to come in this method
    # We can accommodate for multiple visuals if required, barplot, jointplot, scatterplot

    columns = instructions.get("columns") 
    x_axis = columns.get("x")
    y_axis = columns.get("y")
    plot_type = instructions.get("plot_type")
    estimator = instructions.get("estimator", "mean")
    hue_column = instructions.get("hue_column")
    column_div = instructions.get("catplot_col")

    plot_width = instructions.get("plot_size").get("width")
    plot_height = instructions.get("plot_size").get("height")

    if len(columns) > 2:
        raise Exception("Total number of columns cannot exceed 2.")    
    
    if plot_type == "barplot":
        plt.figure(figsize = (plot_width, plot_height))
        sns.barplot(data = data, x = x_axis, y = y_axis, ci = False, estimator=estimator)
    
    elif plot_type == "scatterplot":
        plt.figure(figsize = (plot_width, plot_height))
        sns.scatterplot(data = data, x = x_axis, y = y_axis)
    
    # elif plot_type == "catplot":
    #     plt.figure(figsize = (plot_width, plot_height))
    #     sns.catplot(data = data, x = x_axis, hue = hue_column, col = column_div, estimator=estimator)
    
    # plt.title(f"Bivariate {plot_type} plot for {x_axis} against {y_axis}")
    plt.savefig(f"./plots/bivariate_plots/{plot_type}_{x_axis}_against_{y_axis}.png")
    