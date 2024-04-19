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
#include "tensorflow\c\c_api.h"
#ifdef _MSC_VER
#pragma hdrstop
#endif

#include "mljit.h"

// #define PRINT_MLJIT_LOG

void TensorBufferDeallocator(void* data, size_t a, void* b)
{
    free(data);
}

template <typename T>
T* AddTensorInput(int         NumInputs,
                  TF_Input*   Input,
                  TF_Tensor** InputValues,
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
    assert((dimsNum < 2) || (dimsNum == 2 && dim1 > 0));

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

    assert(inputCount < NumInputs);
    Input[inputCount]       = t;
    InputValues[inputCount] = tensor;
    inputCount++;

    return data;
};

template <typename T>
T* AddTensorOutput(int         NumOutputs,
                   TF_Output*  Output,
                   TF_Tensor** OutputValues,
                   int&        outputCount,
                   TF_Graph*   graph,
                   const char* name,
                   int         index,
                   TF_DataType dtype,
                   int         dimsNum,
                   int64_t     dim0,
                   int64_t     dim1)
{
    assert(dimsNum <= 2 && dimsNum > 0);
    assert(dim0 > 0);
    assert((dimsNum < 2) || (dimsNum == 2 && dim1 > 0));

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

    assert(outputCount < NumOutputs);
    Output[outputCount]       = t;
    OutputValues[outputCount] = tensor;
    outputCount++;

    return data;
};

void mljit_run_cse_policy()
{
    TF_Graph*          graph       = TF_NewGraph();
    TF_Status*         status      = TF_NewStatus();
    TF_SessionOptions* SessionOpts = TF_NewSessionOptions();
    TF_Buffer*         RunOpts     = NULL;

    const char* saved_policy_dir = "C:\\work\\mljit\\saved_policy\\";
    const char* tags             = "serve"; // saved_model_cli

    int         ntags = 1;
    TF_Session* session =
        TF_LoadSessionFromSavedModel(SessionOpts, RunOpts, saved_policy_dir, &tags, ntags, graph, NULL, status);

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

    //****** Get input tensor
    int         NumInputs   = 28;
    TF_Input*   Input       = (TF_Input*)malloc(sizeof(TF_Input) * NumInputs);
    TF_Tensor** InputValues = (TF_Tensor**)malloc(sizeof(TF_Tensor*) * NumInputs);

    int inputCount = 0;

    AddTensorInput<float>(NumInputs, Input, InputValues, inputCount, graph, "action_cse_cost_ex", TF_FLOAT,
                          /* dimsNum */ 1,
                          /* dim0 */ 1, /* dim1 */ 0);
    AddTensorInput<int64_t>(NumInputs, Input, InputValues, inputCount, graph, "action_cse_use_count_weighted_log",
                            TF_INT64,
                            /* dimsNum */ 1, /* dim0 */ 1, /* dim1 */ 0);
    AddTensorInput<int64_t>(NumInputs, Input, InputValues, inputCount, graph, "action_cse_def_count_weighted_log",
                            TF_INT64,
                            /* dimsNum */ 1, /* dim0 */ 1, /* dim1 */ 0);
    AddTensorInput<float>(NumInputs, Input, InputValues, inputCount, graph, "action_cse_cost_sz", TF_FLOAT,
                          /* dimsNum */ 1,
                          /* dim0 */ 1, /* dim1 */ 0);
    AddTensorInput<int64_t>(NumInputs, Input, InputValues, inputCount, graph, "action_cse_use_count", TF_INT64,
                            /* dimsNum */ 1,
                            /* dim0 */ 1, /* dim1 */ 0);
    AddTensorInput<int64_t>(NumInputs, Input, InputValues, inputCount, graph, "action_cse_def_count", TF_INT64,
                            /* dimsNum */ 1,
                            /* dim0 */ 1, /* dim1 */ 0);
    AddTensorInput<int64_t>(NumInputs, Input, InputValues, inputCount, graph, "action_cse_is_live_across_call",
                            TF_INT64,
                            /* dimsNum */ 1, /* dim0 */ 1, /* dim1 */ 0);
    AddTensorInput<int64_t>(NumInputs, Input, InputValues, inputCount, graph, "action_cse_is_int", TF_INT64,
                            /* dimsNum */ 1,
                            /* dim0 */ 1, /* dim1 */ 0);
    AddTensorInput<int64_t>(NumInputs, Input, InputValues, inputCount, graph, "action_cse_is_constant_not_shared",
                            TF_INT64,
                            /* dimsNum */ 1, /* dim0 */ 1, /* dim1 */ 0);
    AddTensorInput<int64_t>(NumInputs, Input, InputValues, inputCount, graph, "action_cse_is_shared_constant", TF_INT64,
                            /* dimsNum */ 1, /* dim0 */ 1, /* dim1 */ 0);
    AddTensorInput<int64_t>(NumInputs, Input, InputValues, inputCount, graph, "action_cse_cost_is_MIN_CSE_COST",
                            TF_INT64,
                            /* dimsNum */ 1, /* dim0 */ 1, /* dim1 */ 0);
    AddTensorInput<int64_t>(NumInputs, Input, InputValues, inputCount, graph, "action_cse_is_constant_live_across_call",
                            TF_INT64,
                            /* dimsNum */ 1, /* dim0 */ 1, /* dim1 */ 0);
    AddTensorInput<int64_t>(NumInputs, Input, InputValues, inputCount, graph, "action_cse_is_constant_min_cost",
                            TF_INT64,
                            /* dimsNum */ 1, /* dim0 */ 1, /* dim1 */ 0);
    AddTensorInput<int64_t>(NumInputs, Input, InputValues, inputCount, graph,
                            "action_cse_cost_is_MIN_CSE_COST_live_across_call", TF_INT64, /* dimsNum */ 1, /* dim0 */ 1,
                            /* dim1 */ 0);
    AddTensorInput<int64_t>(NumInputs, Input, InputValues, inputCount, graph, "action_cse_is_GTF_MAKE_CSE", TF_INT64,
                            /* dimsNum */ 1,
                            /* dim0 */ 1, /* dim1 */ 0);
    AddTensorInput<int64_t>(NumInputs, Input, InputValues, inputCount, graph, "action_cse_num_distinct_locals",
                            TF_INT64,
                            /* dimsNum */ 1, /* dim0 */ 1, /* dim1 */ 0);
    AddTensorInput<int64_t>(NumInputs, Input, InputValues, inputCount, graph, "action_cse_num_local_occurrences",
                            TF_INT64,
                            /* dimsNum */ 1, /* dim0 */ 1, /* dim1 */ 0);
    AddTensorInput<int64_t>(NumInputs, Input, InputValues, inputCount, graph, "action_cse_has_call", TF_INT64,
                            /* dimsNum */ 1,
                            /* dim0 */ 1, /* dim1 */ 0);
    AddTensorInput<int64_t>(NumInputs, Input, InputValues, inputCount, graph,
                            "action_log_cse_use_count_weighted_times_cost_ex", TF_INT64, /* dimsNum */ 1, /* dim0 */ 1,
                            /* dim1 */ 0);
    AddTensorInput<int64_t>(NumInputs, Input, InputValues, inputCount, graph,
                            "action_log_cse_use_count_weighted_times_num_local_occurrences", TF_INT64, /* dimsNum */ 1,
                            /* dim0 */ 1,
                            /* dim1 */ 0);
    AddTensorInput<float>(NumInputs, Input, InputValues, inputCount, graph, "action_cse_distance", TF_FLOAT,
                          /* dimsNum */ 1,
                          /* dim0 */ 1, /* dim1 */ 0);
    AddTensorInput<int64_t>(NumInputs, Input, InputValues, inputCount, graph, "action_cse_is_containable", TF_INT64,
                            /* dimsNum */ 1,
                            /* dim0 */ 1, /* dim1 */ 0);
    AddTensorInput<int64_t>(NumInputs, Input, InputValues, inputCount, graph, "action_cse_is_cheap_containable",
                            TF_INT64,
                            /* dimsNum */ 1, /* dim0 */ 1, /* dim1 */ 0);
    AddTensorInput<int64_t>(NumInputs, Input, InputValues, inputCount, graph,
                            "action_cse_is_live_across_call_in_LSRA_ordering", TF_INT64, /* dimsNum */ 1, /* dim0 */ 1,
                            /* dim1 */ 0);
    AddTensorInput<int64_t>(NumInputs, Input, InputValues, inputCount, graph, "action_log_pressure_estimated_weight",
                            TF_INT64,
                            /* dimsNum */ 1, /* dim0 */ 1, /* dim1 */ 0);
    AddTensorInput<float>(NumInputs, Input, InputValues, inputCount, graph, "action_reward", TF_FLOAT,
                          /* dimsNum */ 1,
                          /* dim0 */ 1,
                          /* dim1 */ 0);
    AddTensorInput<int>(NumInputs, Input, InputValues, inputCount, graph, "action_step_type", TF_INT32,
                        /* dimsNum */ 1,
                        /* dim0 */ 1, /* dim1 */ 0);
    AddTensorInput<float>(NumInputs, Input, InputValues, inputCount, graph, "action_discount", TF_FLOAT,
                          /* dimsNum */ 1,
                          /* dim0 */ 1, /* dim1 */ 0);

    assert(NumInputs == inputCount);

    //********* Get Output tensor
    int         NumOutputs   = 1;
    TF_Output*  Output       = (TF_Output*)malloc(sizeof(TF_Output) * NumOutputs);
    TF_Tensor** OutputValues = (TF_Tensor**)malloc(sizeof(TF_Tensor*) * NumOutputs);

    int outputCount = 0;

    AddTensorOutput<int64_t>(NumOutputs, Output, OutputValues, outputCount, graph, "StatefulPartitionedCall", 0,
                             TF_INT64, 1, 1, 0); // cse_decision

    assert(NumOutputs == outputCount);

    TF_SessionRun(session, nullptr, (TF_Output*)&Input[0], &InputValues[0], NumInputs, &Output[0], &OutputValues[0],
                  NumOutputs, nullptr, 0, nullptr, status);

    if (TF_GetCode(status) == TF_OK)
    {
#ifdef PRINT_MLJIT_LOG
        printf("TF_SessionRun OK\n");
#endif
        auto buff      = (int64_t*)TF_TensorData(OutputValues[0]);
        bool shouldCse = buff[0]; // TODO: Do something with the output.
        //printf("shouldCse: %i\n", shouldCse);
    }
    else
    {
        printf("%s", TF_Message(status));
    }

    // Delete tensors
    for (int i = 0; i < NumInputs; i++)
    {
        TF_DeleteTensor(InputValues[i]);
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

    for (int i = 0; i < NumOutputs; i++)
    {
        TF_DeleteTensor(OutputValues[i]);
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

    // Delete the session options.
    TF_DeleteSessionOptions(SessionOpts);
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

    free(InputValues);
    free(OutputValues);
    free(Input);
    free(Output);
}
