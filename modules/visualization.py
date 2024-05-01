from modules import data
from modules import database
import matplotlib.pyplot as plt
from sklearn import metrics

def create_ROC(database_name, model_id):
    """
    Create and store ROC curve for a given model

    Parameters:
    database_name (str): name of the database
    model_id (int): model id

    Returns:
    file_name (str): name of the file where the ROC curve is stored
    """

    eval = database.retrieve_evaluations(database_name, model_id)
    fp = eval[0][3]
    fn = eval[0][4]
    tp = eval[0][5]
    tn = eval[0][6]

    fpr = fp / (fp + tn)
    tpr = tp / (tp + fn)

    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % metrics.auc(fpr, tpr))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    # Save the figure
    file_name = f"/static/img/graphs/ROC_curve_model_{model_id}.svg"
    plt.savefig(file_name)
    plt.close()

    return file_name

def create_confusion_matrix(database_name, model_id):
    """
    Create and store confusion matrix for a given model

    Parameters:
    database_name (str): name of the database
    model_id (int): model id

    Returns:
    file_name (str): name of the file where the confusion matrix is stored
    """

    eval = database.retrieve_evaluations(database_name, model_id)
    fp = eval[0][3]
    fn = eval[0][4]
    tp = eval[0][5]
    tn = eval[0][6]

    # Plot the confusion matrix
    plt.figure()
    plt.imshow([[tp, fp], [fn, tn]], interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = ['Positive', 'Negative']
    plt.xticks([0, 1], tick_marks)
    plt.yticks([0, 1], tick_marks)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # Save the figure
    file_name = f"/static/img/graphs/confusion_matrix_model_{model_id}.svg"
    plt.savefig(file_name)
    plt.close()

    return file_name

def create_PRC(database_name, model_id):
    """
    Create and store Precision-Recall curve for a given model

    Parameters:
    database_name (str): name of the database
    model_id (int): model id

    Returns:
    file_name (str): name of the file where the PRC curve is stored
    """

    eval = database.retrieve_evaluations(database_name, model_id)
    fp = eval[0][3]
    fn = eval[0][4]
    tp = eval[0][5]
    tn = eval[0][6]

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    # Plot the PRC curve
    plt.figure()
    plt.plot(recall, precision, label='PRC curve')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend(loc="lower right")

    # Save the figure
    file_name = f"/static/img/graphs/PRC_curve_model_{model_id}.svg"
    plt.savefig(file_name)
    plt.close()

    return file_name

