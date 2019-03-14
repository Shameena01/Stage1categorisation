import xml.etree.cElementTree as ET

NodePurityLimit = 0.5
def build_xml_tree__Grad(dt_clf,node_id,node_pos,parent_depth,parent_elementTree,learning_rate):
    """
    Takes a tree given from the ensemble and makes the xml output
    """
    tree = dt_clf.tree_
    n_nodes = tree.node_count
    children_left = tree.children_left
    children_right = tree.children_right
    feature = tree.feature
    threshold = tree.threshold
    value = tree.value
    if (children_left[node_id] != children_right[node_id]):
        node_depth = parent_depth + 1
        if node_id == 0:
            pos = 's'
        else:
            pos = node_pos
        depth = str(node_depth)
        IVar = str(feature[node_id])
        Cut = str(threshold[node_id])
        node_elementTree = ET.SubElement(
            parent_elementTree,"Node",
            pos=pos,
            depth=depth,
            NCoef='0',
            IVar=IVar,
            Cut=Cut,
            cType='1',
            res='0.0e+01',
            rms='0.0e+00',
            purity='0.0e+00',
            nType='0'
        )
        build_xml_tree__Grad(
            dt_clf, children_left[node_id], "l",
            node_depth, node_elementTree, learning_rate
        )
        build_xml_tree__Grad(
            dt_clf, children_right[node_id], "r",
            node_depth, node_elementTree, learning_rate
        )
    else:
        node_depth = parent_depth + 1
        if node_id == 0:
            pos = 's'
        else:
            pos = node_pos
        depth = node_depth
        IVar = -1
        global NodePurityLimit
        sig = value[node_id][0][0]*learning_rate/2.0
        purity = "0.0e+00"

        node_elementTree = ET.SubElement(
            parent_elementTree, "Node", pos=pos,
            depth=str(depth), NCoef="0", IVar=str(IVar),
            Cut="0.0e+00", cType="1", res=str(sig),
            rms="0.0e+00", purity=str(purity), nType="-99"
        )


def convert_bdt__Grad(bdt_clf, input_var_list, tmva_outfile_xml, X_train):

    NTrees = bdt_clf.n_estimators
    learning_rate = bdt_clf.learning_rate
    var_list = input_var_list
    MethodSetup = ET.Element("MethodSetup", Method="BDT::BDT")
    GeneralInfo = ET.SubElement(MethodSetup, "GeneralInfo")
    Info_Creator = ET.SubElement(
        GeneralInfo, "Info", name="Creator", value="VBF-learn (Yacine Haddad)"
    )
    Info_AnalysisType = ET.SubElement(
        GeneralInfo, "Info", name="AnalysisType", value="Classification"
    )

    # <Options>
    Options = ET.SubElement(MethodSetup, "Options")
    Option_NodePurityLimit = ET.SubElement(
        Options, "Option", name="NodePurityLimit", modified="No"
    ).text = str(NodePurityLimit)
    Option_BoostType = ET.SubElement(
        Options, "Option", name="BoostType", modified="Yes"
    ).text = "Grad"
    Option_NTrees = ET.SubElement(
        Options, "Option", name='NTrees', modified="Yes"
    ).text = str(bdt_clf.n_estimators)
    Option_MaxDepth = ET.SubElement(
        Options, "Option", name='MaxDepth', modified="Yes"
    ).text = str(bdt_clf.max_depth)
    Option_Shrinkage = ET.SubElement(
        Options, "Option", name="Shrinkage", modified="Yes"
    ).text = str(bdt_clf.learning_rate)
    Option_UseNvars = ET.SubElement(
        Options, "Option", name='UseNvars', modified="Yes"
    ).text = str(bdt_clf.max_features)

    # <Variables>
    Variables = ET.SubElement(MethodSetup, "Variables", NVar=str(len(var_list)))
    for ind, val in enumerate(var_list):
        max_val = str(X_train[:,ind].max())
        min_val = str(X_train[:,ind].min())
        print(min_val, max_val)
        Variable = ET.SubElement(
            Variables, "Variable", VarIndex=str(ind), Type='F',
            Expression=val, Label=val, Title=val, Unit="", Internal=val,
            Min=min_val, Max=max_val
        )
    # <Spectators>
    Spectators = ET.SubElement(MethodSetup, 'Spectators', NSpec='0')
    # <Classes>
    Classes = ET.SubElement(MethodSetup, "Classes", NClass='2')
    class_Creator = ET.SubElement(Classes, 'class', Name='Signal', Index='0')
    class_Creator = ET.SubElement(Classes, 'class', Name='Background', Index='1')
    # <Transformations>
    Transformations = ET.SubElement(
        MethodSetup, 'Transformations', NTransformations='0'
    )
    # <MVAPdfs>
    MVAPdfs = ET.SubElement(MethodSetup, 'MVAPdfs')
    # <Weights>
    Weights = ET.SubElement(MethodSetup, "Weights", NTrees=str(NTrees), AnalysisType="1")    
    for idx, dt in enumerate(bdt_clf.estimators_[:, 0]):
        BinaryTree = ET.SubElement(Weights, "BinaryTree", type="DecisionTree", boostWeight="1.0e+00", itree=str(idx))
        build_xml_tree__Grad(dt, 0, "s", -1, BinaryTree,learning_rate)

    XMLtext= ET.tostringlist(MethodSetup)
    OutputStr=''
    level=-1

    for t in XMLtext:

        if t[0]=='<':
            level=level+1
        if t[0:2]=='</':
            #remove the added level because of '<'
            level=level-1

        if t[-1]=='"' or t[0]=='<' or t[-2:]=='/>':
            t="\n"+"    "*level+t 

        if t[-2:]=='/>' or t[level*4+1:level*4+3]=='</': #end of block 
            level=level-1

        #if t[-1:]=='>':    
        #    t=t+'\n'

        OutputStr=OutputStr+t

    f = open(tmva_outfile_xml, "w")
    f.write(OutputStr)
    f.close()
