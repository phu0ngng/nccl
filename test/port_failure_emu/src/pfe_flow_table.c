/**
 * @file pfe_flow_table.c
 * @brief Flow table operations for Port Failure Emulation Library
 * 
 * Implements creation and destruction of flow tables, flow groups, and flow table entries.
 */

#include "include/pfe_internal.h"
#include <errno.h>
#include <string.h>

/* ========================================================================== */
/*                       Flow Table Creation                                  */
/* ========================================================================== */

pfe_internal_result_t pfe_create_flow_table(struct pfe_device_ctx *dev_ctx,
                                             struct mlx5dv_devx_obj **ft_obj,
                                             uint32_t *ft_id,
                                             uint32_t table_type,
                                             const char *desc) {
    uint32_t in[DEVX_ST_SZ_DW(create_flow_table_in)] = {0};
    uint32_t out[DEVX_ST_SZ_DW(create_flow_table_out)] = {0};
    void *ft_ctx;

    memset(in, 0, sizeof(in));
    memset(out, 0, sizeof(out));

    ft_ctx = DEVX_ADDR_OF(create_flow_table_in, in, flow_table_context);
    DEVX_SET(create_flow_table_in, in, opcode, MLX5_CMD_OPCODE_CREATE_FLOW_TABLE);
    DEVX_SET(create_flow_table_in, in, table_type, table_type);
    DEVX_SET(flow_table_context, ft_ctx, log_size, PFE_FT_LOG_SIZE);

    *ft_obj = mlx5dv_devx_obj_create(dev_ctx->ibv_ctx, in, sizeof(in), out, sizeof(out));
    if (!*ft_obj) {
        fprintf(stderr, "[PFE ERROR %s] Failed to create %s flow table: %s, syndrome: 0x%x\n",
                __func__, desc, strerror(errno), DEVX_GET(create_flow_table_out, out, syndrome));
        return PFE_INTERNAL_ERROR;
    }

    *ft_id = DEVX_GET(create_flow_table_out, out, table_id);
    // Success - flow table created (no verbose logging in internal functions)
    return PFE_INTERNAL_SUCCESS;
}

/* ========================================================================== */
/*                       Flow Group Creation                                  */
/* ========================================================================== */

pfe_internal_result_t pfe_create_flow_group(struct pfe_device_ctx *dev_ctx,
                                             struct mlx5dv_devx_obj **fg_obj,
                                             uint32_t *fg_id,
                                             uint32_t table_id,
                                             uint32_t table_type,
                                             const char *desc) {
    uint32_t in[DEVX_ST_SZ_DW(create_flow_group_in)] = {0};
    uint32_t out[DEVX_ST_SZ_DW(create_flow_group_out)] = {0};

    // Configure the flow group
    DEVX_SET(create_flow_group_in, in, opcode, MLX5_CMD_OPCODE_CREATE_FLOW_GROUP);
    DEVX_SET(create_flow_group_in, in, table_type, table_type);
    DEVX_SET(create_flow_group_in, in, group_type, MLX5_CREATE_FLOW_GROUP_IN_GROUP_TYPE_TCAM_SUBTABLE);
    DEVX_SET(create_flow_group_in, in, table_id, table_id);
    DEVX_SET(create_flow_group_in, in, start_flow_index, PFE_FG_START_FLOW_IDX);
    DEVX_SET(create_flow_group_in, in, end_flow_index, PFE_FG_END_FLOW_IDX);

    *fg_obj = mlx5dv_devx_obj_create(dev_ctx->ibv_ctx, in, sizeof(in), out, sizeof(out));
    if (!*fg_obj) {
        fprintf(stderr, "[PFE ERROR] Creating %s flow group: %s, syndrome: 0x%x\n",
                desc, strerror(errno), DEVX_GET(create_flow_group_out, out, syndrome));
        return PFE_INTERNAL_ERROR;
    }

    *fg_id = DEVX_GET(create_flow_group_out, out, group_id);
    // Success - flow group created (no verbose logging in internal functions)
    return PFE_INTERNAL_SUCCESS;
}

/* ========================================================================== */
/*                    Flow Table Entry Creation                               */
/* ========================================================================== */

pfe_internal_result_t pfe_create_flow_entry(struct pfe_device_ctx *dev_ctx,
                                             struct mlx5dv_devx_obj ***fte_ptr,
                                             uint32_t fg_id,
                                             uint32_t ft_id,
                                             uint32_t table_type,
                                             int *fte_num,
                                             const char *desc) {
    uint32_t in[DEVX_ST_SZ_DW(set_flow_table_entry_in)] = {0};
    uint32_t out[DEVX_ST_SZ_DW(set_flow_table_entry_out)] = {0};
    void *flow_ctx;

    memset(in, 0, sizeof(in));
    memset(out, 0, sizeof(out));

    // Configure flow table entry
    DEVX_SET(set_flow_table_entry_in, in, opcode, MLX5_CMD_OPCODE_SET_FLOW_TABLE_ENTRY);
    DEVX_SET(set_flow_table_entry_in, in, op_mod, MLX5_SET_FLOW_TABLE_ENTRY_IN_OP_MOD_SET);
    DEVX_SET(set_flow_table_entry_in, in, table_type, table_type);
    DEVX_SET(set_flow_table_entry_in, in, table_id, ft_id);
    DEVX_SET(set_flow_table_entry_in, in, flow_index, 0);

    flow_ctx = DEVX_ADDR_OF(set_flow_table_entry_in, in, flow_context);
    DEVX_SET(flow_context, flow_ctx, group_id, fg_id);
    DEVX_SET(flow_context, flow_ctx, action, MLX5_FLOW_CONTEXT_ACTION_DROP);

    *fte_ptr = malloc(sizeof(struct mlx5dv_devx_obj *));
    if (!*fte_ptr) {
        fprintf(stderr, "[PFE ERROR] Failed to allocate memory for %s flow table entry\n", desc);
        return PFE_INTERNAL_ERROR;
    }

    (*fte_ptr)[0] = mlx5dv_devx_obj_create(dev_ctx->ibv_ctx, in, sizeof(in), out, sizeof(out));
    if (!(*fte_ptr)[0]) {
        fprintf(stderr, "[PFE ERROR] Creating %s flow table entry: %s, syndrome: 0x%x\n",
                desc, strerror(errno), DEVX_GET(set_flow_table_entry_out, out, syndrome));
        free(*fte_ptr);
        return PFE_INTERNAL_ERROR;
    }
    *fte_num += 1;

    // Success - flow entry created (no verbose logging in internal functions)
    return PFE_INTERNAL_SUCCESS;
}

/* ========================================================================== */
/*                    Device-Level Flow Table Setup                           */
/* ========================================================================== */

pfe_internal_result_t pfe_create_device_flow_tables(struct pfe_device_ctx *dev_ctx,
                                                     pfe_mode_t mode) {
    // Create flow tables for RDMA mode
    PFE_CHECK(pfe_create_flow_table(dev_ctx, &dev_ctx->tx_rdma.flow_table, 
                                     &dev_ctx->tx_rdma.flow_table_id,
                                     MLX5_CREATE_FLOW_TABLE_IN_TABLE_TYPE_NIC_TX_RDMA, 
                                     "TX RDMA"));

    PFE_CHECK(pfe_create_flow_table(dev_ctx, &dev_ctx->rx_rdma.flow_table, 
                                     &dev_ctx->rx_rdma.flow_table_id,
                                     MLX5_CREATE_FLOW_TABLE_IN_TABLE_TYPE_NIC_RX_RDMA, 
                                     "RX RDMA"));

    // Create flow tables for non-RDMA (all) mode if required
    if (mode == PFE_MODE_ALL) {
        PFE_CHECK(pfe_create_flow_table(dev_ctx, &dev_ctx->tx.flow_table, 
                                         &dev_ctx->tx.flow_table_id,
                                         MLX5_CREATE_FLOW_TABLE_IN_TABLE_TYPE_NIC_TX, 
                                         "TX ALL"));

        PFE_CHECK(pfe_create_flow_table(dev_ctx, &dev_ctx->rx.flow_table, 
                                         &dev_ctx->rx.flow_table_id,
                                         MLX5_CREATE_FLOW_TABLE_IN_TABLE_TYPE_NIC_RX, 
                                         "RX ALL"));
    }

    return PFE_INTERNAL_SUCCESS;
}

pfe_internal_result_t pfe_create_device_flow_groups(struct pfe_device_ctx *dev_ctx,
                                                     pfe_mode_t mode) {
    // Create flow groups for RDMA mode
    PFE_CHECK(pfe_create_flow_group(dev_ctx, &dev_ctx->tx_rdma.flow_group, 
                                     &dev_ctx->tx_rdma.flow_group_id,
                                     dev_ctx->tx_rdma.flow_table_id, 
                                     MLX5_CREATE_FLOW_TABLE_IN_TABLE_TYPE_NIC_TX_RDMA,
                                     "TX RDMA"));

    PFE_CHECK(pfe_create_flow_group(dev_ctx, &dev_ctx->rx_rdma.flow_group, 
                                     &dev_ctx->rx_rdma.flow_group_id,
                                     dev_ctx->rx_rdma.flow_table_id, 
                                     MLX5_CREATE_FLOW_TABLE_IN_TABLE_TYPE_NIC_RX_RDMA,
                                     "RX RDMA"));

    // Create flow groups for non-RDMA (all) mode if required
    if (mode == PFE_MODE_ALL) {
        PFE_CHECK(pfe_create_flow_group(dev_ctx, &dev_ctx->tx.flow_group, 
                                         &dev_ctx->tx.flow_group_id,
                                         dev_ctx->tx.flow_table_id, 
                                         MLX5_CREATE_FLOW_TABLE_IN_TABLE_TYPE_NIC_TX,
                                         "TX ALL"));

        PFE_CHECK(pfe_create_flow_group(dev_ctx, &dev_ctx->rx.flow_group, 
                                         &dev_ctx->rx.flow_group_id,
                                         dev_ctx->rx.flow_table_id, 
                                         MLX5_CREATE_FLOW_TABLE_IN_TABLE_TYPE_NIC_RX,
                                         "RX ALL"));
    }

    return PFE_INTERNAL_SUCCESS;
}

pfe_internal_result_t pfe_create_device_flow_entries(struct pfe_device_ctx *dev_ctx,
                                                      pfe_mode_t mode) {
    // Create flow table entries for RDMA mode
    PFE_CHECK(pfe_create_flow_entry(dev_ctx, &dev_ctx->tx_rdma.flow_table_entries, 
                                     dev_ctx->tx_rdma.flow_group_id,
                                     dev_ctx->tx_rdma.flow_table_id, 
                                     MLX5_CREATE_FLOW_TABLE_IN_TABLE_TYPE_NIC_TX_RDMA,
                                     &dev_ctx->tx_rdma.flow_table_entry_size, 
                                     "TX RDMA"));

    PFE_CHECK(pfe_create_flow_entry(dev_ctx, &dev_ctx->rx_rdma.flow_table_entries, 
                                     dev_ctx->rx_rdma.flow_group_id,
                                     dev_ctx->rx_rdma.flow_table_id, 
                                     MLX5_CREATE_FLOW_TABLE_IN_TABLE_TYPE_NIC_RX_RDMA,
                                     &dev_ctx->rx_rdma.flow_table_entry_size, 
                                     "RX RDMA"));

    // Create flow table entries for non-RDMA (all) mode if required
    if (mode == PFE_MODE_ALL) {
        PFE_CHECK(pfe_create_flow_entry(dev_ctx, &dev_ctx->tx.flow_table_entries, 
                                         dev_ctx->tx.flow_group_id,
                                         dev_ctx->tx.flow_table_id, 
                                         MLX5_CREATE_FLOW_TABLE_IN_TABLE_TYPE_NIC_TX,
                                         &dev_ctx->tx.flow_table_entry_size, 
                                         "TX ALL"));

        PFE_CHECK(pfe_create_flow_entry(dev_ctx, &dev_ctx->rx.flow_table_entries, 
                                         dev_ctx->rx.flow_group_id,
                                         dev_ctx->rx.flow_table_id, 
                                         MLX5_CREATE_FLOW_TABLE_IN_TABLE_TYPE_NIC_RX,
                                         &dev_ctx->rx.flow_table_entry_size, 
                                         "RX ALL"));
    }

    return PFE_INTERNAL_SUCCESS;
}

/* ========================================================================== */
/*                    Resource Destruction                                    */
/* ========================================================================== */

static void destroy_flow_entries(struct mlx5dv_devx_obj **entries, int size) {
    if (!entries) return;

    for (int i = 0; i < size; i++) {
        if (entries[i]) {
            mlx5dv_devx_obj_destroy(entries[i]);
        }
    }
    free(entries);
}

static void destroy_object(struct mlx5dv_devx_obj *obj) {
    if (obj) {
        mlx5dv_devx_obj_destroy(obj);
    }
}

void pfe_destroy_flow_table_resources(struct pfe_flow_table_resources *resources) {
    if (!resources) return;

    // Destroy flow table entries
    destroy_flow_entries(resources->flow_table_entries, 
                        resources->flow_table_entry_size);
    resources->flow_table_entries = NULL;
    resources->flow_table_entry_size = 0;

    // Destroy flow group
    destroy_object(resources->flow_group);
    resources->flow_group = NULL;

    // Destroy flow table
    destroy_object(resources->flow_table);
    resources->flow_table = NULL;
}

void pfe_destroy_device_resources(struct pfe_device_ctx *dev_ctx,
                                   pfe_mode_t mode) {
    if (!dev_ctx) return;

    // Destroy RDMA resources
    pfe_destroy_flow_table_resources(&dev_ctx->tx_rdma);
    pfe_destroy_flow_table_resources(&dev_ctx->rx_rdma);

    // Destroy non-RDMA resources if mode is MODE_ALL
    if (mode == PFE_MODE_ALL) {
        pfe_destroy_flow_table_resources(&dev_ctx->tx);
        pfe_destroy_flow_table_resources(&dev_ctx->rx);
    }
}
