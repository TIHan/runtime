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

MLJIT_Session::MLJIT_Session()
{
}

void TensorBufferDeallocator(void* data, size_t a, void* b)
{
    free(data);
}

template <typename T>
T* AddTensorInput(int         numInputs,
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

    return data;
};

template <typename T>
T* AddTensorOutput(int         numOutputs,
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

    assert(outputCount < numOutputs);
    output[outputCount]       = t;
    outputValues[outputCount] = tensor;
    outputCount++;

    return data;
};

template <typename T>
T* AddScalarInput(int         numInputs,
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
T* AddScalarOutput(int         numOutputs,
                   TF_Output*  output,
                   TF_Tensor** outputValues,
                   int&        outputCount,
                   TF_Graph*   graph,
                   const char* name,
                   TF_DataType dtype,
                   int         index)
{
    return AddTensorOutput<T>(numOutputs, output, outputValues, outputCount, graph, name, dtype, index, 1, 1, 0);
}

MLJIT_Session_CSE* mljit_session_create_cse()
{
    TF_Graph*          graph       = TF_NewGraph();
    TF_Status*         status      = TF_NewStatus();
    TF_SessionOptions* sessionOpts = TF_NewSessionOptions();

    const char* savedPolicyDir = "C:\\work\\mljit\\saved_policy\\";

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
    }

    //****** Get input tensors
    int         numInputs   = 28;
    TF_Input*   input       = (TF_Input*)malloc(sizeof(TF_Input) * numInputs);
    TF_Tensor** inputValues = (TF_Tensor**)malloc(sizeof(TF_Tensor*) * numInputs);

    int inputCount = 0;

    AddScalarInput<float>(numInputs, input, inputValues, inputCount, graph,
                                                                                    "action_cse_cost_ex", TF_FLOAT);
    AddScalarInput<int64_t>(numInputs, input, inputValues, inputCount, graph,
                                                                                    "action_cse_use_count_weighted_log", TF_INT64);
    AddScalarInput<int64_t>(numInputs, input, inputValues, inputCount, graph,
                                                                                    "action_cse_def_count_weighted_log", TF_INT64);
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
    AddScalarInput<int64_t>(numInputs, input, inputValues, inputCount, graph,
                                                                                    "action_log_cse_use_count_weighted_times_cost_ex", TF_INT64);
    AddScalarInput<int64_t>(numInputs, input, inputValues, inputCount, graph,
                                                                                    "action_log_cse_use_count_weighted_times_num_local_occurrences", TF_INT64);
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

    assert(numInputs == inputCount);

    //********* Get Output tensors
    int         numOutputs   = 1;
    TF_Output*  output       = (TF_Output*)malloc(sizeof(TF_Output) * numOutputs);
    TF_Tensor** outputValues = (TF_Tensor**)malloc(sizeof(TF_Tensor*) * numOutputs);

    int outputCount = 0;

    // This is the "cse_decision".
    AddScalarOutput<int64_t>(numOutputs, output, outputValues, outputCount, graph,
                                                                                    "StatefulPartitionedCall", TF_INT64, 0);

    assert(numOutputs == outputCount);

    //********* MLJIT_Session
    auto mljitSession          = new MLJIT_Session_CSE();
    mljitSession->graph        = graph;
    mljitSession->status       = status;
    mljitSession->sessionOpts  = sessionOpts;
    mljitSession->session      = session;
    mljitSession->numInputs    = numInputs;
    mljitSession->input        = input;
    mljitSession->inputValues  = inputValues;
    mljitSession->numOutputs   = numOutputs;
    mljitSession->output       = output;
    mljitSession->outputValues = outputValues;
    return mljitSession;
}

void mljit_session_destroy(MLJIT_Session* mljitSession)
{
    auto graph       = mljitSession->graph;
    auto status      = mljitSession->status;
    auto sessionOpts = mljitSession->sessionOpts;
    auto session     = mljitSession->session;

    auto numInputs   = mljitSession->numInputs;
    auto input       = mljitSession->input;
    auto inputValues = mljitSession->inputValues;

    auto numOutputs   = mljitSession->numOutputs;
    auto output       = mljitSession->output;
    auto outputValues = mljitSession->outputValues;
    
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
    }

    // Delete the status.
    TF_DeleteStatus(status);

    // cleanup
    free(inputValues);
    free(outputValues);
    free(input);
    free(output);

    delete mljitSession;
}

void mljit_session_action(MLJIT_Session* mljitSession)
{
    auto status  = mljitSession->status;
    auto session = mljitSession->session;

    auto numInputs   = mljitSession->numInputs;
    auto input       = mljitSession->input;
    auto inputValues = mljitSession->inputValues;

    auto numOutputs   = mljitSession->numOutputs;
    auto output       = mljitSession->output;
    auto outputValues = mljitSession->outputValues;

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
    }
}

#endif // DEBUG
