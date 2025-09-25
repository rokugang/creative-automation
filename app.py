"""
Streamlit UI for Creative Automation Platform
Simple, clean interface for demo recording
"""

import streamlit as st
import json
from datetime import datetime
from pathlib import Path
import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.models.campaign import CampaignBrief
from src.core.campaign_processor import CampaignProcessor
from src.agents.campaign_monitor import CampaignMonitor

# Page config
st.set_page_config(
    page_title="Creative Automation Platform",
    page_icon="🎨",
    layout="wide"
)

def init_session_state():
    """Initialize session state."""
    if 'processor' not in st.session_state:
        st.session_state.processor = CampaignProcessor()
    if 'monitor' not in st.session_state:
        st.session_state.monitor = CampaignMonitor()
    if 'processed_campaigns' not in st.session_state:
        st.session_state.processed_campaigns = []

def main():
    """Main application."""
    init_session_state()
    
    # Header
    st.title("Creative Automation Platform")
    st.markdown("**Automated social media campaign generation using GenAI**")
    
    # Main content
    st.markdown("---")
    st.header("Process Campaign")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Campaign Brief (JSON)",
        type=["json"],
        help="Select a JSON file containing the campaign brief"
    )
    
    if uploaded_file:
        # Parse file
        try:
            content = uploaded_file.read()
            brief_data = json.loads(content)
            
            # Show brief
            with st.expander("Campaign Brief Details"):
                st.json(brief_data)
            
            # Process button
            if st.button("Process Campaign", type="primary", use_container_width=True):
                with st.spinner("Generating campaign assets..."):
                    try:
                        # Process campaign
                        brief = CampaignBrief.from_dict(brief_data)
                        result = st.session_state.processor.process(brief)
                        
                        # Success message
                        st.success(f"Campaign {brief.campaign_id} processed successfully!")
                        
                        # Show metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Assets Generated", len(result.assets))
                        with col2:
                            st.metric("Processing Time", f"{result.processing_time:.1f}s")
                        with col3:
                            st.metric("Status", "Complete")
                        
                        # Output info
                        output_dir = f"outputs/campaigns/{brief.campaign_id}"
                        st.info(f"Assets saved to: {output_dir}")
                        
                        # Show output structure
                        if result.output_paths:
                            with st.expander("Output Structure"):
                                st.json(result.output_paths)
                        
                    except Exception as e:
                        st.error(f"Error processing campaign: {e}")
                        st.info("Please ensure API keys are configured in .env file")
        
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON file: {e}")
    else:
        # Instructions
        st.info("Upload a campaign brief JSON file to begin. Example files are in the 'examples' folder.")
        
        # Show sample format
        with st.expander("Example Campaign Brief Format"):
            st.code('''
{
    "campaign_id": "CAMP-001",
    "products": [
        {
            "sku": "PROD-001",
            "name": "Product Name",
            "description": "Product description"
        }
    ],
    "target_markets": [
        {
            "region": "US",
            "language": "en"
        }
    ],
    "aspect_ratios": ["1:1", "9:16", "16:9"],
    "messaging": {
        "headline": "Your Headline Here",
        "body": "Product description text",
        "cta": "Shop Now"
    }
}
            ''', language="json")
    
    # Footer
    st.markdown("---")
    st.caption("Adobe FDE Assessment - Rohit Gangupantulu")

if __name__ == "__main__":
    main()
