import streamlit as st
import os
import numpy
import pandas as pd
from streamlit_image_comparison import image_comparison
from PIL import Image

st.set_page_config(page_title="Image-Comparison Example", layout="centered")

def comparation(log_dir):

    st.markdown('#')
    cp_gt_tab, cp_log_tab = st.tabs(["With ground truth", "with other log"])

    with cp_gt_tab:
        log_choice = st.selectbox(
            "Select log", 
            os.listdir(log_dir), 
            key='log_choice', 
            index=None,
        )

        if log_choice:
            predicted_images_folder = os.path.join(log_dir, log_choice, 'imgs_test_all', 'prediction')
            gt_images_folder = os.path.join(log_dir, log_choice, 'imgs_test_all', 'ground_truth')

            try:
                predicted_image_paths = os.listdir(predicted_images_folder)
                image_metric, image_tab = st.tabs(['Metric', 'Image'])

                with image_metric:
                    history = dict(numpy.load(os.path.join(log_dir, log_choice, 'history.npz')))
                    history_pd = pd.DataFrame.from_dict(history)
                    st.line_chart(history, x='iteration', y=['train_psnr', 'test_psnr'])

                with image_tab:
                    image_choice = st.selectbox(
                        "Select image", 
                        predicted_image_paths, 
                        key='image_choice', 
                        index=None,
                    )

                    if image_choice:
                        predicted_image = Image.open(os.path.join(predicted_images_folder, image_choice))
                        gt_image = Image.open(os.path.join(gt_images_folder, image_choice))

                        image_comparison(
                            img1=gt_image,
                            img2=predicted_image,
                            label1="Ground truth",
                            label2="Prediction",
                            show_labels=True,
                            make_responsive=True,
                            in_memory=True,
                        )        
            except:
                st.error("The predicted image folder doesn't exist", icon="🚨")
    
    with cp_log_tab:

        left_choice_log_cl, right_choice_log_cl = st.columns(2)

        with left_choice_log_cl:
            left_log_choice = st.selectbox(
                "Select log", 
                os.listdir(log_dir), 
                key='left_log_choice', 
                index=None,
            )

        with right_choice_log_cl:
            right_log_choice = st.selectbox(
                "Select log", 
                os.listdir(log_dir), 
                key='right_log_choice', 
                index=None,
            )

        if left_log_choice and right_log_choice:
            
            left_images_folder = os.path.join(log_dir, left_log_choice, 'imgs_test_all', 'prediction')
            right_images_folder = os.path.join(log_dir, right_log_choice, 'imgs_test_all', 'prediction')
            

            try:
                left_images_paths = os.listdir(left_images_folder)
                right_images_paths = os.listdir(right_images_folder)                

                image_metric, image_tab = st.tabs(['Metric', 'Image'])

                with image_metric:
                    plot_dict = {}
                    
                    try:
                        left_history = dict(numpy.load(os.path.join(log_dir, left_log_choice, 'history.npz')))
                        left_history_pd = pd.DataFrame.from_dict(left_history)
                        left_history_pd['cat'] = left_log_choice

                        right_history = dict(numpy.load(os.path.join(log_dir, right_log_choice, 'history.npz')))
                        right_history_pd = pd.DataFrame.from_dict(right_history)
                        right_history_pd['cat'] = right_log_choice

                        plot_pd = pd.concat([right_history_pd, left_history_pd])
                        st.line_chart(plot_pd, x='iteration', y=['test_psnr'], color='cat')
                    except:
                        st.error("History data does not correct format", icon="🚨")

                    

                with image_tab:
                    left_image_col, right_image_col = st.columns(2)
                    
                    left_image_choice = left_image_col.selectbox(
                        "Select image", 
                        os.listdir(left_images_folder), 
                        key='left_image_choice', 
                        index=None,
                    )

                    right_image_choice = right_image_col.selectbox(
                        "Select image", 
                        os.listdir(right_images_folder), 
                        key='right_image_choice', 
                        index=None,
                    )

                    if left_image_choice and right_image_choice:
                        left_image = Image.open(os.path.join(left_images_folder, left_image_choice))
                        right_image = Image.open(os.path.join(right_images_folder, right_image_choice))

                        image_comparison(
                            img1=left_image,
                            img2=right_image,
                            label1=left_log_choice,
                            label2=right_log_choice,
                            show_labels=True,
                            make_responsive=True,
                            in_memory=True,
                        )        
            except:
                    st.error("The predicted image folder doesn't exist", icon="🚨")




                    

