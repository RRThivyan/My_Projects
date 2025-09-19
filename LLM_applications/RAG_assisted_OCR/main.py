import streamlit as st
import os
from tempfile import TemporaryDirectory
from processing import ans  

def main():
    st.set_page_config(page_title="Prescription Reader",
                       page_icon=":medical_symbol:")
    
    st.markdown(
    """
    <h1 style='text-align: center;'>Medical Prescription Reader 🩺</h1>
    """,
    unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Select images for processing:", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        with TemporaryDirectory() as uploads_dir:
            # Generate a unique filename
            filename = uploaded_file.name
            file_path = os.path.join(uploads_dir, filename)

            # Save the uploaded file with a progress indicator
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                st.success(f"Upload of '{filename}' complete!")

            # Process the image and display results
            output, raw = ans(file_path)
            # if output is None:
            #     st.write("Error in generating output.")
                

        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded_file, caption='Original Image', use_column_width=True)
            # fields = output
        # with col2:
            fields = output.split(",")
            # for field in fields:
            #     st.write(field)
        with col2:
            # Text area for user to edit the text output
            cleaned_fields = [line.replace('\n', '').strip() + ',' for line in fields]
            cleaned_text = '\n'.join(cleaned_fields)
            edited_text = st.text_area("Edit the text below:", value=cleaned_text, height=400)
            # edited_text = st.text_area("Edit the text below:", value=fields, height=200)

            # Button to save the edited text
            if st.button("Save Edited Text"):
                # st.session_state.saved_text = edited_text
                #     st.success("Text saved successfully!")
                final_output = edited_text.split(',')
                for i in final_output:
                    st.write(i)
        # with col3:
            # st.write(edited_text)
            # Display the edited and saved text
        # if 'saved_text' in st.session_state:
        #     st.write("Saved Text Output:")
            # st.write(st.session_state.saved_text)
            # sent = st.session_state.saved_text.split(",")
            # for text in sent:
            #     st.write(text)

if __name__ == "__main__":
    main()

