import streamlit as st
import os
import glob
from  utils import * 

def data_visualization(data_dir):         

    # Preprocessing
    # st.divider()
    st.subheader('Get data')

    dataset_option = st.selectbox(
        "Select folder dataset", 
        os.listdir(data_dir), 
        key='dataset_option', 
        index=None,
        placeholder="Dataset"
    )

    if dataset_option:

        dataset_folder = os.path.join(data_dir, dataset_option)
        obj_list = get_folder_names(dataset_folder)
        if len(obj_list) == 0:
            pass
        else:
            dataset_obj_option = st.selectbox(
                "Select dataset object", 
                obj_list, 
                key='dataset_obj_option', 
                index=None,
                placeholder="Object"
            )

            if dataset_obj_option:
                dataset_type_option = st.selectbox(
                    "Select data type", 
                    get_folder_names(os.path.join(dataset_folder, dataset_obj_option)), 
                    key='dataset_type_option', 
                    index=None,
                    placeholder="Type"
                )

                image_list = []
                if dataset_type_option:
                    images_path = os.path.join(data_dir, dataset_option, dataset_obj_option, dataset_type_option)
                    image_list = [os.path.join(images_path, iamge_name) for iamge_name in  os.listdir(images_path)]
                                        
                    col1,col2 = st.columns([0.3, 0.7])

                    if 'counter' not in st.session_state: 
                        st.session_state.counter = 0

                    def showPhoto(next):
                        
                        ## Increments the counter to get next photo
                        if next:
                            st.session_state.counter += 1
                            if st.session_state.counter >= len(image_list):
                                st.session_state.counter = 0
                        else:
                            st.session_state.counter -= 1
                            if st.session_state.counter < 0:
                                st.session_state.counter = len(image_list)-1

                        # Select photo a send it to button
                        photo = image_list[st.session_state.counter]
                        col2.image(photo,caption=photo)

                    # Get list of images in folder
                    col1.subheader("List of images")
                    # col1.write(image_list)


                    with col1:
                        bt_col1, bt_col2 = st.columns(2)
                        show_back_btn = bt_col1.button("Back image", on_click=showPhoto, args=([False]))
                        show_next_btn = bt_col2.button("Next image", on_click=showPhoto, args=([True]))
                        
            