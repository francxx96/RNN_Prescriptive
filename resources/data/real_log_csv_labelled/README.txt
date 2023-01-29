The neural network takes as input XES-formatted files.
Logs are in csv format, you can use Fluxicon Disco to export them in xes.

These are real logs used for testing the neural network,
each log type has its own strings to represent standard XES attributes.
Remember that the neural network wants the positive label expressed with "1" and negative with "0".
Here is the list:

#### Traffic fines settings ####
case_id_col = "Case ID"
activity_col = "Activity"
resource_col = "Resource"
timestamp_col = "Complete Timestamp"
label_col = "label"
pos_label = "deviant"
neg_label = "regular"

#### Sepsis Cases settings ####
case_id_col = "Case ID"
activity_col = "Activity"
resource_col = "org:group"
timestamp_col = "time:timestamp"
label_col = "label"
pos_label = "deviant"
neg_label = "regular"

#### Production log settings ####
case_id_col = "Case ID"
activity_col = "Activity"
resource_col = "Resource"
timestamp_col = "Complete Timestamp"
label_col = "label"
neg_label = "regular"
pos_label = "deviant"

#### BPIC2017 settings ####
case_id_col = "Case ID"
activity_col = "Activity"
resource_col = 'org:resource'
timestamp_col = 'time:timestamp'
label_col = "label"
neg_label = "regular"
pos_label = "deviant"

#### Hospital billing settings ####
case_id_col = "Case ID"
activity_col = "Activity"
resource_col = "Resource"
timestamp_col = "Complete Timestamp"
label_col = "label"
neg_label = "regular"
pos_label = "deviant"

#### BPIC2012 settings ####
case_id_col = "Case ID"
activity_col = "Activity"
resource_col = "Resource"
timestamp_col = "Complete Timestamp"
label_col = "label"
neg_label = "regular"
pos_label = "deviant"

#### BPIC2015 settings ####
case_id_col = "Case ID"
activity_col = "Activity"
resource_col = "org:resource"
timestamp_col = "time:timestamp"
label_col = "label"
pos_label = "deviant"
neg_label = "regular"

#### BPIC2011 settings ####
case_id_col = "Case ID"
activity_col = "Activity code"
resource_col = "Producer code"
timestamp_col = "time:timestamp"
label_col = "label"
pos_label = "deviant"
neg_label = "regular"

