#  # export_path_base = sys.argv[-1]
#  #  export_path = os.path.join(
#  #      tf.compat.as_bytes(export_path_base),
#  #      tf.compat.as_bytes(str(FLAGS.model_version)))
#  #  print('Exporting trained model to', export_path)
#  #  builder = tf.saved_model.builder.SavedModelBuilder(export_path)

#  #  tensor_info_x = tf.saved_model.utils.build_tensor_info(X)
#  #  tensor_info_y = tf.saved_model.utils.build_tensor_info(Z3)

#  #  prediction_signature = (
#  #      tf.saved_model.signature_def_utils.build_signature_def(
#  #          inputs={'images': tensor_info_x},
#  #          outputs={'scores': tensor_info_y},
#  #          method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

#  #  legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
#  #  builder.add_meta_graph_and_variables(
#  #      sess, [tf.saved_model.tag_constants.SERVING],
#  #      signature_def_map={
#  #          tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
#  #              prediction_signature_signature,
#  #      },
#  #      legacy_init_op=legacy_init_op)

#  #  builder.save()

#  #  print('Done exporting!')
# import sys
# import pymysql
# db = pymysql.connect("localhost","root","","attendance_system" )

# # prepare a cursor object using cursor() method
# cursor = db.cursor()
# sql = """INSERT INTO `bsc_csit_7_a`(Date, Name, Auth1, Auth2, Auth3, Auth4, Auth5, Remarks) VALUES ('2054-2-2','akash',1,1,1,1,1,1)"""
# # execute SQL query using execute() method.
# cursor.execute(sql)

# db.commit()

# # disconnect from server
# db.close()
import numpy as np
print(np.maximum(2,3))