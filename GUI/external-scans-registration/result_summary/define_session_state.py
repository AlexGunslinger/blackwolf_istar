import streamlit as st

def define_session_state(Proj,type=None):
    if type == None:
        if Proj.registration_type == 'Internal scans':
            if Proj.floating_roof:
                # define session station values - top    
                if 'plot_metric_top' not in st.session_state:
                    st.session_state.plot_metric_top = False
                if 'sel_slice_top' not in st.session_state:    
                    st.session_state.sel_slice_top = None
                if 'plot_type_top' not in st.session_state:    
                    st.session_state.plot_type_top = False
                if 'sel_metric_top' not in st.session_state:
                    st.session_state.sel_metric_top = None         
                if 'plot_vertical_cross_sec_top' not in st.session_state:
                    st.session_state.plot_vertical_cross_sec_top = False
                if 'plot_horizontal_cross_sec_top' not in st.session_state:
                    st.session_state.plot_horizontal_cross_sec_top = False  
                # define session station values - bot    
                if 'plot_metric_bot' not in st.session_state:
                    st.session_state.plot_metric_bot = False
                if 'sel_slice_bot' not in st.session_state:        
                    st.session_state.sel_slice_bot = None
                if 'plot_type_bot' not in st.session_state:
                    st.session_state.plot_type_bot = False
                if 'sel_metric_bot' not in st.session_state:
                    st.session_state.sel_metric_bot = None     
                if 'plot_horizontal_cross_sec_bot' not in st.session_state:        
                    st.session_state.plot_horizontal_cross_sec_bot = False
                if 'plot_vertical_cross_sec_bot' not in st.session_state:
                    st.session_state.plot_vertical_cross_sec_bot = False
            else:
                # define session station values     
                if 'plot_metric' not in st.session_state:      
                    st.session_state.plot_metric = False
                if 'sel_slice' not in st.session_state:    
                    st.session_state.sel_slice = None    
                if 'plot_type' not in st.session_state:
                    st.session_state.plot_type = False    
                if 'sel_metric' not in st.session_state:
                    st.session_state.sel_metric = None
                if 'plot_vertical_cross_sec' not in st.session_state:
                    st.session_state.plot_vertical_cross_sec = False
                if 'plot_horizontal_cross_sec' not in st.session_state:
                    st.session_state.plot_horizontal_cross_sec = False 
        else:
            # define session station values     
            if 'plot_metric' not in st.session_state:      
                st.session_state.plot_metric = False
            if 'sel_slice' not in st.session_state:    
                st.session_state.sel_slice = None    
            if 'plot_type' not in st.session_state:
                st.session_state.plot_type = False    
            if 'sel_metric' not in st.session_state:
                st.session_state.sel_metric = None
            if 'plot_vertical_cross_sec' not in st.session_state:
                st.session_state.plot_vertical_cross_sec = False
            if 'plot_horizontal_cross_sec' not in st.session_state:
                st.session_state.plot_horizontal_cross_sec = False    

    elif type == 'with floating roof':
        # restart session station values               
        st.session_state.plot_metric = False
        st.session_state.sel_slice = None    
        st.session_state.plot_type = False    
        st.session_state.sel_metric = None
        st.session_state.plot_vertical_cross_sec = False
        st.session_state.plot_horizontal_cross_sec = False  

    elif type == 'without floating roof' or type == 'external':
        st.session_state.plot_metric_top = False    
        st.session_state.plot_type_top = None
        st.session_state.plot_type_top = False
        st.session_state.sel_metric_top = None         
        st.session_state.plot_vertical_cross_sec_top = False
        st.session_state.plot_horizontal_cross_sec_top = False  

        st.session_state.plot_metric_bot = False        
        st.session_state.plot_type_bot = None
        st.session_state.plot_type_bot = False
        st.session_state.sel_metric_bot = None             
        st.session_state.plot_vertical_cross_sec_bot = False
        st.session_state.plot_horizontal_cross_sec_bot = False
            