import streamlit as st
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, AllChem
from mordred import Calculator, descriptors
import pandas as pd
from autogluon.tabular import TabularPredictor
import tempfile
import base64
from io import BytesIO
from autogluon.tabular import FeatureMetadata

# 添加 CSS 样式
st.markdown(
    """
    <style>
    .stApp {
        border: 2px solid #808080;
        border-radius: 20px;
        margin: 50px auto;
        max-width: 39%; /* 设置最大宽度 */
        background-color: #f9f9f9f9;
        padding: 20px; /* 增加内边距 */
        box-sizing: border-box;
    }
    .rounded-container h2 {
        margin-top: -80px;
        text-align: center;
        background-color: #e0e0e0e0;
        padding: 10px;
        border-radius: 10px;
    }
    .rounded-container blockquote {
        text-align: left;
        margin: 20px auto;
        background-color: #f0f0f0;
        padding: 10px;
        font-size: 1.1em;
        border-radius: 10px;
    }
    a {
        color: #0000EE;
        text-decoration: underline;
    }
    .process-text, .molecular-weight {
        font-family: Arial, sans-serif;
        font-size: 16px;
        color: #333;
    }
    .stDataFrame {
        margin-top: 10px;
        margin-bottom: 0px !important;
    }
    .molecule-img {
        display: block;
        margin: 20px auto;
        max-width: 300px;
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 5px;
        background-color: white;
    }
    .molecule-svg {
        background-color: transparent; /* 设置背景为透明 */
        display: block;
        margin: 20px auto;
        max-width: 300px;
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 5px;
    }
    /* 针对小屏幕的优化 */
    @media (max-width: 768px) {
        .rounded-container {
            padding: 10px; /* 减少内边距 */
        }
        .rounded-container blockquote {
            font-size: 0.9em; /* 缩小字体 */
        }
        .rounded-container h2 {
            font-size: 1.2em; /* 调整标题字体大小 */
        }
        .stApp {
            padding: 1px !important; /* 减少内边距 */
            max-width: 99%; /* 设置最大宽度 */
        }
        .process-text, .molecular-weight {
            font-size: 0.9em; /* 缩小文本字体 */
        }
        .molecule-img, .molecule-svg {
            max-width: 200px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 页面标题和简介
st.markdown(
    """
    <div class='rounded-container'>
        <h2>Predict Organic Fluorescence <br>Emission Wavelengths</h2>
        <blockquote>
            1. This website aims to quickly predict the emission wavelength of organic molecules based on their structure (SMILES) and solvent using machine learning models.<br>
            2. Code and data are available at <a href='<url id="d0vf3vv6rtpeof5sdre0" type="url" status="parsed" title="GitHub - dqzs/Fluorescence-Emission-Wavelength-Prediction: Autogluon was used to train the model and make predictions on the molecule" wc="842">https://github.com/dqzs/Fluorescence-Emission-Wavelength-Prediction</url> ' target='_blank'><url id="d0vf3vv6rtpeof5sdre0" type="url" status="parsed" title="GitHub - dqzs/Fluorescence-Emission-Wavelength-Prediction: Autogluon was used to train the model and make predictions on the molecule" wc="842">https://github.com/dqzs/Fluorescence-Emission-Wavelength-Prediction</url> </a>.
        </blockquote>
    </div>
    """,
    unsafe_allow_html=True,
)

# 溶剂数据字典
solvent_data = {
    # 溶剂字典内容保持不变
}

# 溶剂选择下拉菜单
solvent = st.selectbox("Select Solvent:", list(solvent_data.keys()))

# SMILES 输入区域
smiles = st.text_input("Enter the SMILES representation of the molecule:", placeholder="e.g., NC1=CC=C(C=C1)C(=O)O")

# 提交按钮
submit_button = st.button("Submit and Predict", key="predict_button")

# 指定的描述符列表
required_descriptors = ["MAXdssC", "VSA_EState7", "SMR_VSA10", "PEOE_VSA8"]

# 缓存模型加载器以避免重复加载
@st.cache_resource(show_spinner=False)
def load_predictor():
    """缓存模型加载，避免重复加载导致内存溢出"""
    return TabularPredictor.load("./ag-20250529_123557")

def mol_to_image(mol, size=(300, 300)):
    """将分子转换为透明背景的SVG图像"""
    d2d = Draw.MolDraw2DSVG(size[0], size[1])
    d2d.drawOptions().background = None  # 设置背景为透明
    d2d.DrawMolecule(mol)
    d2d.FinishDrawing()
    svg = d2d.GetDrawingText()
    return svg

def get_descriptors(mol):
    """获取指定的分子描述符"""
    # 计算RDKit描述符
    try:
        rdkit_descs = {
            "VSA_EState7": Descriptors.VSA_EState7(mol),
            "SMR_VSA10": Descriptors.SMR_VSA10(mol),
            "PEOE_VSA8": Descriptors.PEOE_VSA8(mol),
        }
    except:
        # 如果计算失败，使用默认值
        rdkit_descs = {
            "VSA_EState7": 0.0,
            "SMR_VSA10": 0.0,
            "PEOE_VSA8": 0.0,
        }

    # 计算Mordred描述符
    try:
        calc = Calculator(descriptors, ignore_3D=True)
        mordred_desc = calc(mol)
        maxdssc = mordred_desc["MAXdssC"] if "MAXdssC" in mordred_desc else 0.0
    except:
        maxdssc = 0.0

    return {
        "MAXdssC": maxdssc,
        **rdkit_descs
    }

# 如果点击提交按钮
if submit_button:
    if not smiles:
        st.error("Please enter a valid SMILES string.")
    elif not solvent:
        st.error("Please select a solvent.")
    else:
        with st.spinner("Processing molecule and making predictions..."):
            try:
                # 处理SMILES输入
                st.markdown('<div class="process-text">Processing SMILES input...</div>', unsafe_allow_html=True)
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    # 添加H原子并生成2D坐标
                    mol = Chem.AddHs(mol)
                    AllChem.Compute2DCoords(mol)

                    # 显示分子结构
                    svg = mol_to_image(mol)
                    # 使用自定义CSS类确保背景一致
                    st.markdown(f'<div class="molecule-svg">{svg}</div>', unsafe_allow_html=True)

                    # 计算分子量
                    mol_weight = Descriptors.MolWt(mol)
                    st.markdown(f'<div class="molecular-weight">Molecular Weight: {mol_weight:.2f} g/mol</div>',
                                unsafe_allow_html=True)

                    # 获取溶剂参数
                    solvent_params = solvent_data[solvent]

                    # 计算指定描述符
                    st.info("Calculating molecular descriptors...")
                    desc_values = get_descriptors(mol)

                    # 创建输入数据表
                    input_data = {
                        "SMILES": [smiles],
                        "Et30": [solvent_params["Et30"]],
                        "SP": [solvent_params["SP"]],
                        "SdP": [solvent_params["SdP"]],
                        "SA": [solvent_params["SA"]],
                        "SB": [solvent_params["SB"]],
                        "MAXdssC": [desc_values["MAXdssC"]],
                        "VSA_EState7": [desc_values["VSA_EState7"]],
                        "SMR_VSA10": [desc_values["SMR_VSA10"]],
                        "PEOE_VSA8": [desc_values["PEOE_VSA8"]],
                        "image": ["Molecular Structure"]
                    }

                    input_df = pd.DataFrame(input_data)
                    
                    # 显示输入数据
                    st.write("Input Data:")
                    st.dataframe(input_df)

                    # 创建预测用数据框
                    predict_data = {
                        "SMILES": [smiles],
                        "Et30": [solvent_params["Et30"]],
                        "SP": [solvent_params["SP"]],
                        "SdP": [solvent_params["SdP"]],
                        "SA": [solvent_params["SA"]],
                        "SB": [solvent_params["SB"]],
                        "MAXdssC": [desc_values["MAXdssC"]],
                        "VSA_EState7": [desc_values["VSA_EState7"]],
                        "SMR_VSA10": [desc_values["SMR_VSA10"]],
                        "PEOE_VSA8": [desc_values["PEOE_VSA8"]]
                    }
                    
                    predict_df = pd.DataFrame(predict_data)
                    
                    # 加载模型并预测
                    st.info("Loading the model and predicting the emission wavelength...")
                    try:
                        # 使用缓存的模型加载方式
                        predictor = load_predictor()
                        
                        # 指定模型列表
                        model_options = ['LightGBM',
                                         'LightGBMXT',
                                         'CatBoost',
                                         'XGBoost',
                                         'NeuralNetTorch',
                                         'LightGBMLarge',
                                         'MultiModalPredictor',
                                         'WeightedEnsemble_L2'
                                        ]
                        predict_df_1 = pd.concat([predict_df,predict_df],axis=0)
                        # 获取预测结果
                        predictions_dict = {}
                        for model in model_options:
                            try:
                                predictions = predictor.predict(predict_df_1, model=model)
                                predictions_dict[model] = predictions.astype(int).apply(lambda x: f"{x} nm")
                            except Exception as model_error:
                                st.warning(f"Model {model} prediction failed: {str(model_error)}")
                                predictions_dict[model] = "Error"

                        # 显示预测结果
                        st.write("Prediction Results:")
                        st.markdown(
                            "**Note:** WeightedEnsemble_L2 is a meta-model combining predictions from other models.")
                        results_df = pd.DataFrame(predictions_dict)
                        st.dataframe(results_df.iloc[:1,:])

                    except Exception as e:
                        st.error(f"Model loading failed: {str(e)}")

                else:
                    st.error("Invalid SMILES input. Please check the format.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
