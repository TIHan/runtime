// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

/*XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
XX                                                                           XX
XX                              MLJIT                                        XX
XX                                                                           XX
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
*/

#include "jitpch.h"
#include "jitstd/algorithm.h"
#ifdef _MSC_VER
#pragma hdrstop
#endif

#ifdef DEBUG

#include "mljit.h"

// #define PRINT_MLJIT_LOG

MLJIT_Policy::MLJIT_Policy()
{
}

void MLJIT_Policy::Action()
{
    auto status  = this->status;
    auto session = this->session;

    auto numInputs   = this->numInputs;
    auto input       = this->input;
    auto inputValues = this->inputValues;

    auto numOutputs   = this->numOutputs;
    auto output       = this->output;
    auto outputValues = this->outputValues;

    TF_SessionRun(session, nullptr, (TF_Output*)&input[0], &inputValues[0], numInputs, &output[0], &outputValues[0],
                  numOutputs, nullptr, 0, nullptr, status);

    if (TF_GetCode(status) == TF_OK)
    {
#ifdef PRINT_MLJIT_LOG
        printf("TF_SessionRun OK\n");
#endif
    }
    else
    {
        printf("%s", TF_Message(status));
        assert(!"Failed TF_SessionRun");
    }
}

void TensorBufferDeallocator(void* data, size_t a, void* b)
{
    free(data);
}

template <typename T>
void AddTensorInput(int         numInputs,
                    TF_Input*   input,
                    TF_Tensor** inputValues,
                    int&        inputCount,
                    TF_Graph*   graph,
                    const char* name,
                    TF_DataType dtype,
                    int         dimsNum,
                    int64_t     dim0,
                    int64_t     dim1)
{
    assert(dimsNum <= 2 && dimsNum > 0);
    assert(dim0 > 0);
    assert((dimsNum < 2 && dim1 == 0) || (dimsNum == 2 && dim1 > 0));

    TF_Input t = {TF_GraphOperationByName(graph, name), 0};
    if (t.oper == NULL)
    {
        printf("ERROR: Failed TF_GraphOperationByName '%s'\n", name);
        assert(!"Failed TF_GraphOperationByName");
    }
    else
    {
#ifdef PRINT_MLJIT_LOG
        printf("TF_GraphOperationByName '%s' is OK\n", name);
#endif
    }

    size_t byteSize = 0;
    if (dimsNum == 1)
    {
        byteSize = sizeof(T) * dim0;
    }
    else if (dimsNum == 2)
    {
        byteSize = sizeof(T) * dim0 * dim1;
    }
    else
    {
        assert(!"no way");
    }

    T* data = (T*)calloc(1, byteSize);

    int64_t dims[] = {dim0, dim1};

    TF_Tensor* tensor = TF_NewTensor(dtype, dims, dimsNum, data, byteSize, &TensorBufferDeallocator, NULL);
    if (tensor != NULL)
    {
#ifdef PRINT_MLJIT_LOG
        printf("TF_NewTensor is OK\n");
#endif
    }
    else
    {
        printf("ERROR: Failed TF_NewTensor\n");
    }

    assert(inputCount < numInputs);
    input[inputCount]       = t;
    inputValues[inputCount] = tensor;
    inputCount++;
};

template <typename T>
void AddTensorOutput(int         numOutputs,
                     TF_Output*  output,
                     TF_Tensor** outputValues,
                     int&        outputCount,
                     TF_Graph*   graph,
                     const char* name,
                     TF_DataType dtype,
                     int         index,
                     int         dimsNum,
                     int64_t     dim0,
                     int64_t     dim1)
{
    assert(dimsNum <= 2 && dimsNum > 0);
    assert(dim0 > 0);
    assert((dimsNum < 2 && dim1 == 0) || (dimsNum == 2 && dim1 > 0));

    TF_Output t = {TF_GraphOperationByName(graph, name), index};
    if (t.oper == NULL)
    {  
        printf("ERROR: Failed TF_GraphOperationByName '%s'\n", name);
        assert(!"Failed TF_GraphOperationByName");
    }
    else
    {
#ifdef PRINT_MLJIT_LOG
        printf("TF_GraphOperationByName '%s' is OK\n", name);
#endif
    }

    size_t byteSize = 0;
    if (dimsNum == 1)
    {
        byteSize = sizeof(T) * dim0;
    }
    else if (dimsNum == 2)
    {
        byteSize = sizeof(T) * dim0 * dim1;
    }
    else
    {
        assert(!"no way");
    }

    T* data = (T*)calloc(1, byteSize);

    int64_t dims[] = {dim0, dim1};

    TF_Tensor* tensor = TF_NewTensor(dtype, dims, dimsNum, data, byteSize, &TensorBufferDeallocator, NULL);
    if (tensor != NULL)
    {
#ifdef PRINT_MLJIT_LOG
        printf("TF_NewTensor is OK\n");
#endif
    }
    else
    {
        printf("ERROR: Failed TF_NewTensor\n");
        assert(!"Failed TF_NewTensor");
    }

    assert(outputCount < numOutputs);
    output[outputCount]       = t;
    outputValues[outputCount] = tensor;
    outputCount++;
};

template <typename T>
void AddScalarInput(int         numInputs,
                    TF_Input*   input,
                    TF_Tensor** inputValues,
                    int&        inputCount,
                    TF_Graph*   graph,
                    const char* name,
                    TF_DataType dtype)
{
    return AddTensorInput<T>(numInputs, input, inputValues, inputCount, graph, name, dtype, 1, 1, 0);
}

template <typename T>
void AddScalarOutput(int         numOutputs,
                     TF_Output*  output,
                     TF_Tensor** outputValues,
                     int&        outputCount,
                     TF_Graph*   graph,
                     const char* name,
                     TF_DataType dtype,
                     int         index)
{
    AddTensorOutput<T>(numOutputs, output, outputValues, outputCount, graph, name, dtype, index, 1, 1, 0);
}

void Add_CategoricalProjectionNetwork_logits_Output(
    int numOutputs, TF_Output* output, TF_Tensor** outputValues, int& outputCount, TF_Graph* graph)
{
    AddTensorOutput<float>(numOutputs, output, outputValues, outputCount, graph, "StatefulPartitionedCall",
                           TF_FLOAT, 0, 2, 1, 2);
}

void mljit_add_cse_policy_inputs(
    int numInputs, TF_Input* input, TF_Tensor** inputValues, int& inputCount, TF_Graph* graph)
{
    AddScalarInput<int64_t>(numInputs, input, inputValues, inputCount, graph,
                                                                                    "action_cse_index", TF_INT64);
    AddScalarInput<float>(numInputs, input, inputValues, inputCount, graph,
                                                                                    "action_cse_cost_ex", TF_FLOAT);
    AddScalarInput<float>(numInputs, input, inputValues, inputCount, graph,
                                                                                    "action_cse_use_count_weighted_log", TF_FLOAT);
    AddScalarInput<float>(numInputs, input, inputValues, inputCount, graph,
                                                                                    "action_cse_def_count_weighted_log", TF_FLOAT);
    AddScalarInput<float>(numInputs, input, inputValues, inputCount, graph,
                                                                                    "action_cse_cost_sz", TF_FLOAT);
    AddScalarInput<int64_t>(numInputs, input, inputValues, inputCount, graph,
                                                                                    "action_cse_use_count", TF_INT64);
    AddScalarInput<int64_t>(numInputs, input, inputValues, inputCount, graph,
                                                                                    "action_cse_def_count", TF_INT64);
    AddScalarInput<int64_t>(numInputs, input, inputValues, inputCount, graph,
                                                                                    "action_cse_is_live_across_call", TF_INT64);
    AddScalarInput<int64_t>(numInputs, input, inputValues, inputCount, graph,
                                                                                    "action_cse_is_int", TF_INT64);
    AddScalarInput<int64_t>(numInputs, input, inputValues, inputCount, graph,
                                                                                    "action_cse_is_constant_not_shared", TF_INT64);
    AddScalarInput<int64_t>(numInputs, input, inputValues, inputCount, graph,
                                                                                    "action_cse_is_shared_constant", TF_INT64);
    AddScalarInput<int64_t>(numInputs, input, inputValues, inputCount, graph,
                                                                                    "action_cse_cost_is_MIN_CSE_COST", TF_INT64);
    AddScalarInput<int64_t>(numInputs, input, inputValues, inputCount, graph,
                                                                                    "action_cse_is_constant_live_across_call", TF_INT64);
    AddScalarInput<int64_t>(numInputs, input, inputValues, inputCount, graph,
                                                                                    "action_cse_is_constant_min_cost", TF_INT64);
    AddScalarInput<int64_t>(numInputs, input, inputValues, inputCount, graph,
                                                                                    "action_cse_cost_is_MIN_CSE_COST_live_across_call", TF_INT64);
    AddScalarInput<int64_t>(numInputs, input, inputValues, inputCount, graph,
                                                                                    "action_cse_is_GTF_MAKE_CSE", TF_INT64);
    AddScalarInput<int64_t>(numInputs, input, inputValues, inputCount, graph,
                                                                                    "action_cse_num_distinct_locals", TF_INT64);
    AddScalarInput<int64_t>(numInputs, input, inputValues, inputCount, graph,
                                                                                    "action_cse_num_local_occurrences", TF_INT64);
    AddScalarInput<int64_t>(numInputs, input, inputValues, inputCount, graph,
                                                                                    "action_cse_has_call", TF_INT64);
    AddScalarInput<float>(numInputs, input, inputValues, inputCount, graph,
                                                                                    "action_log_cse_use_count_weighted_times_cost_ex", TF_FLOAT);
    AddScalarInput<float>(numInputs, input, inputValues, inputCount, graph,
                                                                                    "action_log_cse_use_count_weighted_times_num_local_occurrences", TF_FLOAT);
    AddScalarInput<float>(numInputs, input, inputValues, inputCount, graph,
                                                                                    "action_cse_distance", TF_FLOAT);
    AddScalarInput<int64_t>(numInputs, input, inputValues, inputCount, graph,
                                                                                    "action_cse_is_containable", TF_INT64);
    AddScalarInput<int64_t>(numInputs, input, inputValues, inputCount, graph,
                                                                                    "action_cse_is_cheap_containable", TF_INT64);
    AddScalarInput<int64_t>(numInputs, input, inputValues, inputCount, graph,
                                                                                    "action_cse_is_live_across_call_in_LSRA_ordering", TF_INT64);
    AddScalarInput<int64_t>(numInputs, input, inputValues, inputCount, graph,
                                                                                    "action_log_pressure_estimated_weight", TF_INT64);
    AddScalarInput<float>(numInputs, input, inputValues, inputCount, graph,
                                                                                    "action_reward", TF_FLOAT);
    AddScalarInput<int>(numInputs, input, inputValues, inputCount, graph,
                                                                                    "action_step_type", TF_INT32);
    AddScalarInput<float>(numInputs, input, inputValues, inputCount, graph,
                                                                                    "action_discount", TF_FLOAT);
}

MLJIT_CsePolicyBase* mljit_create_cse_policy(const char* savedPolicyDir)
{
    TF_Graph*          graph       = TF_NewGraph();
    TF_Status*         status      = TF_NewStatus();
    TF_SessionOptions* sessionOpts = TF_NewSessionOptions();

    int         ntags = 1;
    const char* tags  = "serve";

    TF_Session* session =
        TF_LoadSessionFromSavedModel(sessionOpts, NULL, savedPolicyDir, &tags, ntags, graph, NULL, status);

    if (TF_GetCode(status) == TF_OK)
    {
#ifdef PRINT_MLJIT_LOG
        printf("TF_LoadSessionFromSavedModel OK\n");
#endif
    }
    else
    {
        printf("%s", TF_Message(status));
        assert(!"Failed TF_LoadSessionFromSavedModel");
    }

    //****** Get input tensors
    int         numInputs   = 29;
    TF_Input*   input       = (TF_Input*)malloc(sizeof(TF_Input) * numInputs);
    TF_Tensor** inputValues = (TF_Tensor**)malloc(sizeof(TF_Tensor*) * numInputs);

    int inputCount = 0;

    mljit_add_cse_policy_inputs(numInputs, input, inputValues, inputCount, graph);

    assert(numInputs == inputCount);

    //********* Get Output tensors
    int         numOutputs   = 1;
    TF_Output*  output       = (TF_Output*)malloc(sizeof(TF_Output) * numOutputs);
    TF_Tensor** outputValues = (TF_Tensor**)malloc(sizeof(TF_Tensor*) * numOutputs);

    int outputCount = 0;

    // This is the "cse_decision".
    AddScalarOutput<int64_t>(numOutputs, output, outputValues, outputCount, graph, "StatefulPartitionedCall", TF_INT64,
                             0);

    assert(numOutputs == outputCount);

    //********* MLJIT_Policy
    auto policy          = new MLJIT_CsePolicy();
    policy->graph        = graph;
    policy->status       = status;
    policy->sessionOpts  = sessionOpts;
    policy->session      = session;
    policy->numInputs    = numInputs;
    policy->input        = input;
    policy->inputValues  = inputValues;
    policy->numOutputs   = numOutputs;
    policy->output       = output;
    policy->outputValues = outputValues;
    return policy;
}

MLJIT_CsePolicyBase* mljit_create_cse_collect_policy(const char* savedPolicyDir)
{
    TF_Graph*          graph       = TF_NewGraph();
    TF_Status*         status      = TF_NewStatus();
    TF_SessionOptions* sessionOpts = TF_NewSessionOptions();

    int         ntags = 1;
    const char* tags  = "serve";

    TF_Session* session =
        TF_LoadSessionFromSavedModel(sessionOpts, NULL, savedPolicyDir, &tags, ntags, graph, NULL, status);

    if (TF_GetCode(status) == TF_OK)
    {
#ifdef PRINT_MLJIT_LOG
        printf("TF_LoadSessionFromSavedModel OK\n");
#endif
    }
    else
    {
        printf("%s", TF_Message(status));
        assert(!"Failed TF_LoadSessionFromSavedModel");
    }

    //****** Get input tensors
    int         numInputs   = 29;
    TF_Input*   input       = (TF_Input*)malloc(sizeof(TF_Input) * numInputs);
    TF_Tensor** inputValues = (TF_Tensor**)malloc(sizeof(TF_Tensor*) * numInputs);

    int inputCount = 0;

    mljit_add_cse_policy_inputs(numInputs, input, inputValues, inputCount, graph);

    assert(numInputs == inputCount);

    //********* Get Output tensors
    int         numOutputs   = 2;
    TF_Output*  output       = (TF_Output*)malloc(sizeof(TF_Output) * numOutputs);
    TF_Tensor** outputValues = (TF_Tensor**)malloc(sizeof(TF_Tensor*) * numOutputs);

    int outputCount = 0;

    // This is only for the 'collect_policy'.
    Add_CategoricalProjectionNetwork_logits_Output(numOutputs, output, outputValues, outputCount, graph);

    // This is the "cse_decision".
    AddScalarOutput<int64_t>(numOutputs, output, outputValues, outputCount, graph, "StatefulPartitionedCall", TF_INT64,
                             1);

    assert(numOutputs == outputCount);

    //********* MLJIT_Policy
    auto policy          = new MLJIT_CseCollectPolicy();
    policy->graph        = graph;
    policy->status       = status;
    policy->sessionOpts  = sessionOpts;
    policy->session      = session;
    policy->numInputs    = numInputs;
    policy->input        = input;
    policy->inputValues  = inputValues;
    policy->numOutputs   = numOutputs;
    policy->output       = output;
    policy->outputValues = outputValues;
    return policy;
}

MLJIT_CsePolicyBase* mljit_try_create_cse_policy()
{
    const char* mljitEnabled = getenv("DOTNET_MLJitEnabled");
    if (!mljitEnabled || (strcmp(mljitEnabled, "1") != 0))
        return nullptr;

    const char* savedPolicyDir = getenv("DOTNET_MLJitSavedPolicyPath");

    if (!savedPolicyDir)
        return nullptr;

    return mljit_create_cse_policy(savedPolicyDir);
}

MLJIT_CsePolicyBase* mljit_try_create_cse_collect_policy()
{
    const char* mljitEnabled = getenv("DOTNET_MLJitEnabled");
    if (!mljitEnabled || (strcmp(mljitEnabled, "1") != 0))
        return nullptr;

    const char* savedPolicyDir = getenv("DOTNET_MLJitSavedCollectPolicyPath");

    if (!savedPolicyDir)
        return nullptr;

    return mljit_create_cse_collect_policy(savedPolicyDir);
}

void mljit_destroy_policy(MLJIT_Policy* policy)
{
    auto graph       = policy->graph;
    auto status      = policy->status;
    auto sessionOpts = policy->sessionOpts;
    auto session     = policy->session;

    auto numInputs   = policy->numInputs;
    auto input       = policy->input;
    auto inputValues = policy->inputValues;

    auto numOutputs   = policy->numOutputs;
    auto output       = policy->output;
    auto outputValues = policy->outputValues;
    
    // Delete tensors
    for (int i = 0; i < numInputs; i++)
    {
        TF_DeleteTensor(inputValues[i]);
        if (TF_GetCode(status) == TF_OK)
        {
#ifdef PRINT_MLJIT_LOG
            printf("TF_DeleteTensor OK\n");
#endif
        }
        else
        {
            printf("%s", TF_Message(status));
            assert(!"Failed TF_DeleteTensor");
        }
    }

    for (int i = 0; i < numOutputs; i++)
    {
        TF_DeleteTensor(outputValues[i]);
        if (TF_GetCode(status) == TF_OK)
        {
#ifdef PRINT_MLJIT_LOG
            printf("TF_DeleteTensor OK\n");
#endif
        }
        else
        {
            printf("%s", TF_Message(status));
            assert(!"Failed TF_DeleteTensor");
        }
    }

    // Close the session.
    TF_CloseSession(session, status);
    if (TF_GetCode(status) == TF_OK)
    {
#ifdef PRINT_MLJIT_LOG
        printf("TF_CloseSession OK\n");
#endif
    }
    else
    {
        printf("%s", TF_Message(status));
        assert(!"Failed TF_CloseSession");
    }

    // Delete the session.
    TF_DeleteSession(session, status);
    if (TF_GetCode(status) == TF_OK)
    {
#ifdef PRINT_MLJIT_LOG
        printf("TF_DeleteSession OK\n");
#endif
    }
    else
    {
        printf("%s", TF_Message(status));
        assert(!"Failed TF_DeleteSession");
    }

    // Delete the session options.
    TF_DeleteSessionOptions(sessionOpts);
    if (TF_GetCode(status) == TF_OK)
    {
#ifdef PRINT_MLJIT_LOG
        printf("TF_DeleteSessionOptions OK\n");
#endif
    }
    else
    {
        printf("%s", TF_Message(status));
        assert(!"Failed TF_DeleteSessionOptions");
    }

    // Delete the graph.
    TF_DeleteGraph(graph);
    if (TF_GetCode(status) == TF_OK)
    {
#ifdef PRINT_MLJIT_LOG
        printf("TF_DeleteGraph OK\n");
#endif
    }
    else
    {
        printf("%s", TF_Message(status));
        assert(!"Failed TF_Message");
    }

    // Delete the status.
    TF_DeleteStatus(status);

    // cleanup
    free(inputValues);
    free(outputValues);
    free(input);
    free(output);

    delete policy;
}

#endif // DEBUG
