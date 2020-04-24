score = Pickled_LR_Model.score(Xtest, Ytest)
# Print the Score
print("Test score: {0:.2f} %".format(100 * score))
